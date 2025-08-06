from diffusers.utils.torch_utils import randn_tensor
from tasks.base import BaseGuider
from utils.configs import Arguments
import torch

class BaseGuidance:

    def __init__(self, args: Arguments, custom_noise_fn=None, noise_fn=None):

        self.args = args
        self.guider = BaseGuider(args)
        # Handle both custom_noise_fn and noise_fn parameters for backward compatibility
        if noise_fn is not None:
            self.noise_fn = noise_fn
        elif custom_noise_fn is not None:
            self.noise_fn = custom_noise_fn
        else:
            self.generator = torch.manual_seed(self.args.seed)
            def default_noise_fn (x, sigma, **kwargs):
                noise =  randn_tensor(x.shape, generator=self.generator, device=self.args.device, dtype=x.dtype)
                return sigma * noise + x
            self.noise_fn = default_noise_fn

    def guide_step(
        self,
        x: torch.Tensor,
        t: int,
        unet: torch.nn.Module,
        ts: torch.LongTensor,
        alpha_prod_ts: torch.Tensor,
        alpha_prod_t_prevs: torch.Tensor,
        eta: float,
        **kwargs,
    ) -> torch.Tensor:

        alpha_prod_t = alpha_prod_ts[t]
        alpha_prod_t_prev = alpha_prod_t_prevs[t]
        t = ts[t]

        for recur_step in range(self.args.recur_steps):
    
            eps = unet(x, t)

            # predict x0 using xt and epsilon
            x0 = self._predict_x0(x, eps, alpha_prod_t, **kwargs)

            x_prev = self._predict_x_prev_from_zero(
                x, x0, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs
            )

            x = self._predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs)
        
        return x_prev


    def _predict_x_prev_from_zero(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        alpha_prod_t: torch.Tensor,
        alpha_prod_t_prev: torch.Tensor,
        eta: float,
        t: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        new_epsilon = (
            (xt - alpha_prod_t ** (0.5) * x0) / (1 - alpha_prod_t) ** (0.5)
        )

        return self._predict_x_prev_from_eps(xt, new_epsilon, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)


    def _predict_x_prev_from_eps(
        self,
        xt: torch.Tensor,
        eps: torch.Tensor,
        alpha_prod_t: torch.Tensor,
        alpha_prod_t_prev: torch.Tensor,
        eta: float,
        t: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        sigma = eta * (
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        ) ** (0.5)

        pred_sample_direction = (1 - alpha_prod_t_prev - sigma**2) ** (0.5) * eps
        pred_x0_direction = (xt - (1 - alpha_prod_t) ** (0.5) * eps) / (alpha_prod_t ** (0.5))

        # Equation (12) in DDIM sampling
        mean_pred = alpha_prod_t_prev ** (0.5) * pred_x0_direction + pred_sample_direction

        # Add noise following DiT's exact implementation
        if eta > 0:
            # Create nonzero_mask exactly like DiT
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1)))
            )  # no noise when t == 0
            
            # Generate noise and add it
            noise = torch.randn_like(xt)
            # sigma와 nonzero_mask를 xt와 같은 디바이스로 이동
            sigma = sigma.to(xt.device)
            nonzero_mask = nonzero_mask.to(xt.device)
            mean_pred = mean_pred + nonzero_mask * sigma * noise
        
        return mean_pred


    def _predict_xt(
        self,
        x_prev: torch.Tensor,
        alpha_prod_t: torch.Tensor,
        alpha_prod_t_prev: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        
        xt_mean = (alpha_prod_t / alpha_prod_t_prev) ** (0.5) * x_prev

        return self.noise_fn(xt_mean, (1 - alpha_prod_t / alpha_prod_t_prev) ** (0.5), **kwargs)

        noise = randn_tensor(
            x_prev.shape, generator=self.generator, device=self.args.device, dtype=x_prev.dtype
        )   

        return xt_mean + (1 - alpha_prod_t / alpha_prod_t_prev) ** (0.5) * noise


    def _predict_x0(
        self, xt: torch.Tensor, eps: torch.Tensor, alpha_prod_t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        
        pred_x0 = (xt - (1 - alpha_prod_t) ** (0.5) * eps) / (alpha_prod_t ** (0.5))

        if self.args.clip_x0:
            pred_x0 = torch.clamp(pred_x0, -self.args.clip_sample_range, self.args.clip_sample_range)
        
        return pred_x0