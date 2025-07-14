from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor

import math
import torch
from functools import partial
import numpy as np

from tasks.utils import rescale_grad


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    if arr is None:
        raise ValueError("arr cannot be None in _extract_into_tensor")
    if broadcast_shape is None or not (isinstance(broadcast_shape, (tuple, list)) and isinstance(broadcast_shape[0], int)):
        raise ValueError("broadcast_shape must be a tuple/list of ints in _extract_into_tensor")
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    arr = arr.to(device=timesteps.device)
    if isinstance(timesteps, int):
        timesteps = torch.full((broadcast_shape[0],), timesteps, device=arr.device, dtype=torch.long)
    elif torch.is_tensor(timesteps) and timesteps.dim() == 0:
        timesteps = timesteps.expand(broadcast_shape[0])
    res = arr[timesteps]
    if not torch.is_tensor(res):
        res = torch.tensor(res, device=arr.device)
    res = res.float()
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)
    return res.expand(broadcast_shape)


class TFGGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(TFGGuidance, self).__init__(args, **kwargs)
        self.device = args.device
        
        # Add classifier guidance attributes
        self.classifier = getattr(args, 'classifier', None)
        self.classifier_guidance_scale = getattr(args, 'classifier_guidance_scale', 0.0)
        
        # Add diffusion schedule attributes (needed for DiT-style sampling)
        self.alphas_cumprod = getattr(args, 'alphas_cumprod', None)
        self.alphas_cumprod_prev = getattr(args, 'alphas_cumprod_prev', None)
        
        # If alphas_cumprod is not provided, create a simple linear schedule
        if self.alphas_cumprod is None:
            num_timesteps = getattr(args, 'num_timesteps', 1000)
            beta_start = getattr(args, 'beta_start', 0.0001)
            beta_end = getattr(args, 'beta_end', 0.02)
            
            betas = torch.linspace(beta_start, beta_end, num_timesteps, device=self.device, dtype=torch.float32)
            alphas = 1.0 - betas
            self.alphas_cumprod = torch.cumprod(alphas, dim=0)
            self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device, dtype=torch.float32), self.alphas_cumprod[:-1]])
        
        # Pre-compute DiT-style constants (exactly like DiT)
        if self.alphas_cumprod is not None:
            self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod.cpu().numpy())
            self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod.cpu().numpy() - 1)

    @torch.enable_grad()
    def tilde_get_guidance(self, x0, mc_eps, return_logp=False, **kwargs):
        # x0는 반드시 leaf tensor이고 requires_grad=True여야 함
        x0 = x0.clone().detach().requires_grad_(True)
        flat_x0 = (x0[None] + mc_eps).reshape(-1, *x0.shape[1:])
        outs = self.guider.get_guidance(flat_x0, return_logp=True, check_grad=False, **kwargs)
        batch_size = x0.shape[0]
        avg_logprobs = torch.logsumexp(outs.reshape(mc_eps.shape[0], batch_size), dim=0) - math.log(mc_eps.shape[0])
        if return_logp:
            return avg_logprobs
        _grad = torch.autograd.grad(avg_logprobs.sum(), x0, retain_graph=True)[0]
        _grad = rescale_grad(_grad, clip_scale=self.args.clip_scale, **kwargs)
        return _grad
    
    def get_noise(self, std, shape, eps_bsz=4, **kwargs):
        if std == 0.0:
            return torch.zeros((1, *shape), device=self.device)
        return torch.stack([self.noise_fn(torch.zeros(shape, device=self.device), std, **kwargs) for _ in range(eps_bsz)]) 
    # randn_tensor((4, *shape), device=self.device, generator=self.generator) * std
    
    def get_rho(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.rho_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.rho * scheduler[t] * len(scheduler) / scheduler.sum()

    def get_mu(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.mu_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.mu *  scheduler[t] * len(scheduler) / scheduler.sum()
    
    def get_std(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.sigma_schedule == 'decrease':    # beta_t
            scheduler = (1 - alpha_prod_ts) ** 0.5
        elif self.args.sigma_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.sigma *  scheduler[t]

    def guide_step(
        self,
        x: torch.Tensor,
        t: int,
        transformer: torch.nn.Module,
        ts: torch.LongTensor,
        alpha_prod_ts: torch.Tensor,
        alpha_prod_t_prevs: torch.Tensor,
        eta: float,
        class_labels: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        # Ensure input tensor is float32
        x = x.to(dtype=torch.float32)
        B = x.shape[0]
        # t를 항상 LongTensor(batch 차원 포함)로 변환
        if isinstance(t, int):
            t_tensor = torch.full((B,), t, device=x.device, dtype=torch.long)
        elif torch.is_tensor(t) and t.dim() == 0:
            t_tensor = t.expand(B)
        else:
            t_tensor = t
        # DiT-style: Get model prediction (epsilon or x0)
        model_kwargs = dict(y=class_labels) if class_labels is not None else {}
        model_output = transformer(x, t_tensor, **model_kwargs)
        # DiT with learn_sigma=True outputs 8 channels (epsilon + sigma)
        if model_output.shape[1] == 8:
            eps = model_output[:, :4]
        elif model_output.shape[1] == 4:
            eps = model_output
        else:
            raise ValueError(f"Unexpected model output channels: {model_output.shape[1]}, expected 4 or 8")
        # DiT predicts epsilon, convert to x0 prediction
        pred_xstart = self._predict_x0_from_eps(x, t_tensor, eps, None)
        # Classifier guidance (생략, 기존대로)
        # DiT-style: Compute mean and variance for sampling
        model_mean, model_variance, model_log_variance = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t_tensor
        )
        # DiT-style: Add noise for sampling (exactly like DiT's p_sample)
        # noise 추가 위치 (DiT 공식)
        # t는 항상 (B,) Tensor임
        # t_tensor는 항상 torch.Tensor (B,)로 보장
        if isinstance(t, bool) or isinstance(t, int):
            t_tensor = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)
        else:
            t_tensor = t
        # noise: t=0이면 0, 아니면 randn
        if torch.is_tensor(t_tensor) and (t_tensor > 0).any():
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        # nonzero_mask: (B, 1, 1, 1) shape으로 broadcasting
        if torch.is_tensor(t_tensor):
            nonzero_mask = (t_tensor != 0).float().view(-1, *([1] * (x.dim() - 1)))
        else:
            nonzero_mask = torch.ones_like(x)
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample

    def _predict_x0_from_eps(self, x_t, t, eps, _):
        # Use DiT's pre-computed constants
        sqrt_recip_alphas_cumprod = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod = _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * eps

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """Convert x0 prediction to epsilon prediction (exactly like DiT's _predict_eps_from_xstart)"""
        # Use DiT's pre-computed constants
        sqrt_recip_alphas_cumprod = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod = _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        # DiT's formula: eps = (sqrt_recip_alphas_cumprod * x_t - pred_xstart) / sqrt_recipm1_alphas_cumprod
        return (sqrt_recip_alphas_cumprod * x_t - pred_xstart) / sqrt_recipm1_alphas_cumprod

    def q_posterior_mean_variance(self, x_start, x_t, t):
        # 스케줄 파라미터 준비
        if not hasattr(self, 'betas'):
            args = getattr(self, 'args', None)
            num_timesteps = getattr(args, 'num_timesteps', 1000) if args else 1000
            beta_start = getattr(args, 'beta_start', 0.0001) if args else 0.0001
            beta_end = getattr(args, 'beta_end', 0.02) if args else 0.02
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=x_t.device, dtype=torch.float32)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.alphas_cumprod_prev = torch.cat([
                torch.ones(1, device=x_t.device, dtype=torch.float32),
                self.alphas_cumprod[:-1]
            ])
        # t shape: (B,)
        B = x_t.shape[0]
        if isinstance(t, int):
            t = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        elif torch.is_tensor(t) and t.dim() == 0:
            t = t.expand(B)
        # extract schedule
        betas_t = _extract_into_tensor(self.betas, t, x_t.shape)
        alphas_t = _extract_into_tensor(self.alphas, t, x_t.shape)
        alphas_cumprod_t = _extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
        alphas_cumprod_prev_t = _extract_into_tensor(self.alphas_cumprod_prev, t, x_t.shape)
        # posterior 공식 (DiT)
        posterior_variance = betas_t * (1. - alphas_cumprod_prev_t) / (1. - alphas_cumprod_t)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        posterior_log_variance = torch.log(posterior_variance)
        # mean 공식 (DiT)
        model_mean = (
            torch.sqrt(alphas_cumprod_prev_t) * betas_t * x_start +
            torch.sqrt(alphas_t) * (1. - alphas_cumprod_prev_t) * x_t
        ) / (1. - alphas_cumprod_t)
        return model_mean, posterior_variance, posterior_log_variance