from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
import torch as th
import enum
import math
from diffusion.transformer.openai import ModelMeanType, ModelVarType, LossType
import torch.nn.functional as F
import torchvision.transforms as T
import os
import torch.nn as nn
from typing import List, Union, Any
from PIL import Image
from diffusion.base import BaseSampler
from utils.configs import Arguments

# === UViT용 SDE/ScoreModel/샘플 step 함수 ===
import torch
import numpy as np
from sde import VPSDE, ScoreModel, ReverseSDE
from libs.uvit import UViT

def stp(s, ts: torch.Tensor):
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts

def duplicate(tensor, *size):
    return tensor.unsqueeze(dim=0).expand(*size, *tensor.shape)

class UViT_VPSDE(VPSDE):
    def __init__(self, beta_min=0.1, beta_max=20):
        super().__init__(beta_min=beta_min, beta_max=beta_max)

class UViT_ScoreModel(ScoreModel):
    def __init__(self, nnet: torch.nn.Module, pred: str, sde: UViT_VPSDE, T=1):
        super().__init__(nnet=nnet, pred=pred, sde=sde, T=T)

class UViT_ReverseSDE(ReverseSDE):
    def __init__(self, score_model):
        super().__init__(score_model)

@torch.no_grad()
def uvit_step(x, t, model, model_name_or_path, image_size, **kwargs):
    # UViT 모델 로드
    uvit_model = UViT(img_size=image_size, patch_size=2, embed_dim=512, depth=12, num_heads=8).to(x.device)
    state_dict = torch.load(model_name_or_path, map_location=x.device)
    uvit_model.load_state_dict(state_dict)
    uvit_model.eval()
    sde_obj = UViT_VPSDE(beta_min=0.1, beta_max=20)
    score_model = UViT_ScoreModel(uvit_model, pred='noise_pred', sde=sde_obj)
    rsde = UViT_ReverseSDE(score_model)
    # Euler-Maruyama step (한 스텝만)
    sample_steps = kwargs.get('sample_steps', 1000)
    eps = 1e-3
    T = 1
    timesteps = np.append(0., np.linspace(eps, T, sample_steps))
    timesteps = torch.tensor(timesteps).to(x)
    # t는 step 인덱스, t_idx
    t_idx = t[0].item() if isinstance(t, torch.Tensor) else int(t)
    s = timesteps[t_idx]
    t_val = timesteps[t_idx+1] if t_idx+1 < len(timesteps) else 0.0
    drift = rsde.drift(x, t_val)
    diffusion = rsde.diffusion(t_val)
    dt = s - t_val
    mean = x + drift * dt
    sigma = diffusion * (-dt).sqrt()
    x_next = mean + stp(sigma, torch.randn_like(x)) if s != 0 else mean
    return {"sample": x_next}

def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {desired_count} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            ModelMeanType.EPSILON if not predict_xstart else ModelMeanType.START_X
        ),
        model_var_type=(
            (
                ModelVarType.FIXED_LARGE
                if not sigma_small
                else ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
    )

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_timesteps = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_timesteps] = np.linspace(beta_start, beta_end, warmup_timesteps, dtype=np.float64)
    return betas

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class GaussianDiffusion:
    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None):
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

class _WrappedModel:
    def __init__(self, model, timestep_map, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        return self.model(x, new_ts, **kwargs)

class SpacedDiffusion(GaussianDiffusion):
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t

# DPSGuidance 클래스는 methods/dps.py에서 import하여 사용
# 중복된 DPSGuidance 클래스 제거 - 실제로는 methods/dps.py의 DPSGuidance가 사용됨

import functools
import math
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm
from typing import List, Any, Union, Tuple
import numpy as np
import os
import torch.nn as nn

from diffusers.utils.torch_utils import randn_tensor

from utils.configs import Arguments
from .base import BaseSampler
from methods.base import BaseGuidance
import logger

from tasks.networks.qm9 import dataset
from tasks.networks.qm9.utils import compute_mean_mad
from tasks.networks.qm9.datasets_config import get_dataset_info
from tasks.networks.qm9.models import DistributionProperty, DistributionNodes
from tasks.networks.egnn.EDM import EDM
from tasks.networks.egnn.EGNN import EGNN_dynamics_QM9
from tasks.networks.egnn.utils import assert_correctly_masked, assert_mean_zero_with_mask


class ImageSampler(BaseSampler):

    def __init__(self, args: Arguments):
        super(ImageSampler, self).__init__(args)
        self.args = args
        self.inference_steps = args.inference_steps
        self.device = args.device
        self.target = args.target
        self.object_size = (3, args.image_size, args.image_size)
        self.eta = args.eta
        self.log_traj = args.log_traj
        self.generator = torch.manual_seed(self.seed)
        
        # model_type 결정
        if hasattr(args, 'model_type') and args.model_type is not None:
            self.model_type = args.model_type
        else:
            # 기존 로직: model_name_or_path 기반 추정
            if 'uvit' in args.model_name_or_path.lower():
                self.model_type = 'uvit'
            elif 'transformer' in args.model_name_or_path.lower():
                self.model_type = 'transformer'
            else:
                self.model_type = 'unet'
        
        self.model_name_or_path = args.model_name_or_path
        self._build_diffusion(args)

    def _build_diffusion(self, args):
        '''
            Different diffusion models should be registered here
        '''
        if self.model_type == 'uvit':
            # UViT 모델 로드
            from libs.uvit import UViT
            self.image_size = 32  # UViT용 image_size 설정
            self.model = UViT(
                img_size=32,
                patch_size=2,
                in_chans=3,
                embed_dim=512,
                depth=12,
                num_heads=8,
                mlp_ratio=4,
                qkv_bias=False,
                mlp_time_embed=False,
                num_classes=-1,
                norm_layer=nn.LayerNorm,
                use_checkpoint=False
            ).to(args.device)
            
            # checkpoint 로드
            checkpoint_path = args.model_name_or_path
            if not os.path.exists(checkpoint_path):
                models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', checkpoint_path)
                if os.path.exists(models_path):
                    checkpoint_path = models_path
                else:
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path} or {models_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            
            # UViT SDE 설정
            from sde import VPSDE, ScoreModel, ReverseSDE
            self.sde = VPSDE()
            self.score_model = ScoreModel(self.model, pred='noise_pred', sde=self.sde)
            self.rsde = ReverseSDE(self.score_model)
            self.uvit_sample_steps = args.inference_steps
            def uvit_sample_fn(sample_size, guidance):
                n = sample_size
                device = self.device
                # U-ViT SDE 기반 sampling에 맞는 표준 정규분포 초기 노이즈
                x = torch.randn(n, 3, self.image_size, self.image_size, device=device)
                
                # target 값을 class_labels로 변환
                class_labels = None
                if hasattr(self, 'args') and hasattr(self.args, 'target') and self.args.target is not None:
                    try:
                        target_class = int(self.args.target)
                        class_labels = torch.full((n,), target_class, device=device, dtype=torch.long)
                    except (ValueError, TypeError):
                        pass
                
                for t_idx in reversed(range(self.inference_steps)):
                    t = torch.full((n,), t_idx, device=device, dtype=torch.long)
                    guidance_scale_value = getattr(self.args, 'guidance_scale', 1.0)
                    start_gradient_value = getattr(self.args, 'start_gradient', None)
                    print(f"[DEBUG] guidance_scale from args: {guidance_scale_value}, start_gradient: {start_gradient_value}")
                    x = guidance.guide_step(
                        x, t, self.model, None, None, None, 0.0,
                        class_labels=class_labels, cfg_scale=0.0, diffusion=None,
                        model_type='uvit', model_name_or_path=self.model_name_or_path, image_size=self.image_size,
                        guidance_scale=guidance_scale_value, start_gradient=start_gradient_value
                    )
                return x
            self.sample = uvit_sample_fn
            
        elif self.model_type == 'transformer':
            # DiT (Transformer) 모델 로드
            from .transformer.openai import create_diffusion, DiT_models
            from diffusers.models import AutoencoderKL
            
            latent_size = args.image_size // 8
            self.model = DiT_models['DiT-XL/2'](
                input_size=latent_size,
                num_classes=1000
            ).to(args.device)
            self.model.initialize_weights()
            
            # VAE 로드
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(args.device)
            
            # checkpoint 로드
            checkpoint_path = args.model_name_or_path
            if not os.path.exists(checkpoint_path):
                models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', checkpoint_path)
                if os.path.exists(models_path):
                    checkpoint_path = models_path
                else:
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path} or {models_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            
            # Diffusion 설정
            self.diffusion = create_diffusion(timestep_respacing=str(args.inference_steps))
            
        else:
            # UNet 모델 로드 (기본)
            if 'openai' in args.model_name_or_path:
                from .unet.openai import get_diffusion
            else: 
                from .unet.huggingface import get_diffusion
            
            self.unet, self.ts, self.alpha_prod_ts, self.alpha_prod_t_prevs = get_diffusion(args)

    @torch.no_grad()
    def sample(self, sample_size: int, guidance: BaseGuidance):
        # UViT는 자체 sample 함수를 사용
        if self.model_type == 'uvit':
            return self.sample(sample_size, guidance)
        
        # UNet과 Transformer는 통합된 샘플링 로직 사용
        tot_samples = []
        n_batchs = math.ceil(sample_size / self.per_sample_batch_size)

        for batch_id in range(n_batchs):
            
            self.args.batch_id = batch_id

            if self.model_type == 'transformer':
                # Transformer 샘플링
                batch_size = min(self.per_sample_batch_size, sample_size - batch_id * self.per_sample_batch_size)
                x = torch.randn(batch_size, 4, self.args.image_size // 8, self.args.image_size // 8, device=self.device)
                
                for t_idx in reversed(range(self.inference_steps)):
                    t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
                    x = guidance.guide_step(
                        x, t, self.model, None, None, None, self.eta,
                        model_type='transformer', model_name_or_path=self.model_name_or_path, 
                        image_size=self.args.image_size, diffusion=self.diffusion
                    )
                    
                    if self.log_traj:
                        logger.log_samples(self.tensor_to_obj(x), fname=f'traj/time={t_idx}')
                
                # VAE decode
                x = self.vae.decode(x / 0.18215, return_dict=False)[0]
                
            else:
                # UNet 샘플링 (기본)
                batch_size = min(self.per_sample_batch_size, sample_size - batch_id * self.per_sample_batch_size)
                x = randn_tensor(
                    shape=(batch_size, *self.object_size),
                    generator=self.generator,
                    device=self.device,
                )

                for t in tqdm(range(self.inference_steps), total=self.inference_steps):
                    
                    x = guidance.guide_step(
                        x, t, self.unet,
                        self.ts,
                        self.alpha_prod_ts, 
                        self.alpha_prod_t_prevs,
                        self.eta,
                        model_type='unet'
                    )

                    if self.log_traj:
                        logger.log_samples(self.tensor_to_obj(x), fname=f'traj/time={t}')

            # None 체크 추가
            if x is None:
                print(f"[ERROR] guide_step returned None for batch {batch_id}")
                raise ValueError(f"guide_step returned None for batch {batch_id}")
            
            tot_samples.append(x)
        
        # tot_samples에 None이 있는지 확인
        for i, sample in enumerate(tot_samples):
            if sample is None:
                print(f"[ERROR] tot_samples[{i}] is None")
                raise ValueError(f"tot_samples[{i}] is None")
        
        return torch.concat(tot_samples)

    @staticmethod
    def tensor_to_obj(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            # Convert from [-1, 1] to [0, 1]
            x = (x + 1) / 2
            x = x.clamp(0, 1)
            # Convert to PIL images
            images = []
            for i in range(x.shape[0]):
                img = x[i].permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                images.append(Image.fromarray(img))
            return images
        else:
            return x

    @staticmethod
    def obj_to_tensor(objs: List[Image.Image]) -> torch.Tensor:
        '''
            convert a list of PIL images into tensors
        '''
        images = [to_tensor(pil_image) for pil_image in objs]
        tensor_images = torch.stack(images)
        return tensor_images * 2 - 1


class MoleculeSampler(BaseSampler):
    """
        This class is responsible for sampling molecules using DDIM.

    """
    def __init__(self, args: Arguments):

        super(MoleculeSampler, self).__init__(args)
        self.object_size = None                         # size of tensor shape
        self.inference_steps = args.inference_steps     # number of steps to run the diffusion model
        self.eta = args.eta                             # eta in DDIM paper
        self.log_traj = args.log_traj
        self.generator = torch.manual_seed(self.seed)
        self.per_sample_batch_size = args.per_sample_batch_size

        # prepare unet, prev_t, alpha_prod, alpha_prod_prev...
        self._build_diffusion(args)


    @staticmethod
    def _get_dataloader(args_gen):
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen)
        return dataloaders

    @staticmethod
    def _get_generator(model_path, dataloaders, device, args, property_norms):
        dataset_info = get_dataset_info(args.args_gen.dataset, args.args_gen.remove_h)
        model, nodes_dist, prop_dist = MoleculeSampler._get_model(
            args.args_gen, device, dataset_info, dataloaders['train'],args.target)
        model_state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(model_state_dict)

        if prop_dist is not None:
            prop_dist.set_normalizer(property_norms)
        return model.to(device), nodes_dist, prop_dist, dataset_info

    @staticmethod
    def _get_model(args, device, dataset_info, dataloader_train, target):
        histogram = dataset_info['n_nodes']
        in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
        # in_node_nf: the numbder of atom type
        nodes_dist = DistributionNodes(histogram)

        # pass corresponding property into the pre-defined function
        prop_dist = DistributionProperty(dataloader_train, [target]) if target else None
        
        dynamics_in_node_nf = in_node_nf + 1
    
        net_dynamics = EGNN_dynamics_QM9(
            in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
            n_dims=3, device=device, hidden_nf=args.nf,
            act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
            attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
            inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
            normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)

        vdm = EDM(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges,
        )

        return vdm, nodes_dist, prop_dist

    def _build_diffusion(self, args):
        '''
            Different diffusion models should be registered here
        '''
        # dataloader
        args.args_gen.load_charges = False
        dataloaders = self._get_dataloader(args.args_gen)
        property_norms = compute_mean_mad(dataloaders, [args.target], args.args_gen.dataset)
        # mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']

        # load conditional EDM and property prediction model
        edm, nodes_dist, prop_dist, dataset_info = self._get_generator(
            args.generators_path, dataloaders,
            args.device, args, property_norms,
        )
        self.dataset_info = dataset_info
        self.device = args.device
        self.args_gen = args.args_gen
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist
        self.target = args.target

        self.ts, self.alpha_prod_ts, self.alpha_prod_t_prevs = edm.get_scheduler_params(args)
        self.diffusion: EDM = edm

    def remove_mean_with_mask(self, x, node_mask):
        # masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
        # assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'

        N = node_mask.sum(1, keepdims=True)

        mean = torch.sum(x, dim=1, keepdim=True) / N
        x = x - mean * node_mask
        return x

    def noise_fn(self, x, sigma, node_mask, **kwargs):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
            assert len(size) == 3
            x = torch.randn(size, device=device)

            x_masked = x * node_mask

            # This projection only works because Gaussian is rotation invariant around
            # zero and samples are independent!
            x_projected = self.remove_mean_with_mask(x_masked, node_mask)
            return x_projected

        def sample_gaussian_with_mask(size, device, node_mask):
            x = torch.randn(size, device=device)
            x_masked = x * node_mask
            return x_masked


        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(x.shape[0], x.shape[1], self.diffusion.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = sample_gaussian_with_mask(
            size=(x.shape[0], x.shape[1], self.diffusion.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)

        zs = z * sigma + x
        return torch.cat([
            self.remove_mean_with_mask(zs[:, :, :self.diffusion.n_dims],node_mask),
            zs[:, :, self.diffusion.n_dims:]], dim=2
        )

    @torch.no_grad()
    def sample(self, sample_size: int, guidance: BaseGuidance):

        tot_samples = []
        n_batchs = math.ceil(sample_size / self.per_sample_batch_size)

        for _ in range(n_batchs):

            nodesxsample = self.nodes_dist.sample(self.per_sample_batch_size)
            context = self.prop_dist.sample_batch(nodesxsample).to(self.device)

            max_n_nodes = self.dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

            assert int(torch.max(nodesxsample)) <= max_n_nodes
            batch_size = len(nodesxsample)

            node_mask = torch.zeros(batch_size, max_n_nodes)
            for i in range(batch_size):
                node_mask[i, 0:nodesxsample[i]] = 1

            # Compute edge_mask

            edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
            edge_mask *= diag_mask
            edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(self.device)
            node_mask = node_mask.unsqueeze(2).to(self.device)

            # # TODO FIX: This conditioning just zeros.
            # if context is None:
            #     context = self.prop_dist.sample_batch(nodesxsample)
            context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(self.device) * node_mask

            n_samples = batch_size
            n_nodes = max_n_nodes

            z = self.diffusion.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)
            noise_pred_func = functools.partial(self.diffusion.forward, node_mask=node_mask, edge_mask=edge_mask)

            for t in tqdm(range(self.inference_steps), total=self.inference_steps):
                
                z = guidance.guide_step(
                    z, t, noise_pred_func,
                    self.ts,
                    self.alpha_prod_ts, 
                    self.alpha_prod_t_prevs,
                    self.eta,
                    node_mask=node_mask,
                    edge_mask=edge_mask,
                    target=context,
                )
                # z = z * node_mask
                x = z[..., :self.diffusion.n_dims]
                # project x back to the mean zero space, using the mean computed only on the nodes
                # x = x - x.sum(dim=1, keepdim=True) / node_mask.sum(dim=1, keepdim=True) * node_mask
                # assert_correctly_masked(x, node_mask)
                # assert_mean_zero_with_mask(x, node_mask)

                # we may want to log some trajs
                if self.log_traj:
                    logger.log_samples(self.tensor_to_obj(z), fname=f'traj/time={t}')

            nan_mask = torch.isnan(z.reshape(z.shape[0], -1)).any(dim=1)
            logger.log(f"generate {nan_mask.shape[0]} samples, {nan_mask.sum()} of them are NaNs.")
            if nan_mask.float().mean() >= 0.5:
                logger.log(f'Warning: {nan_mask.float().mean()} of the samples are NaNs.')
                raise ValueError('Too many NaNs in the samples. Drop the run')
            z, node_mask, context = z[~nan_mask], node_mask[~nan_mask], context[~nan_mask]
            edge_mask = edge_mask.view(-1, max_n_nodes, max_n_nodes, 1)[~nan_mask].view(-1, 1)
            x, h = self.diffusion.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=False)

            # x = x - x.sum(dim=1, keepdim=True) / node_mask.sum(dim=1, keepdim=True) * node_mask

            # assert_correctly_masked(x, node_mask)
            # assert_mean_zero_with_mask(x, node_mask)

            one_hot = h['categorical']
            charges = h['integer']

            # assert_correctly_masked(one_hot.float(), node_mask)
            # if self.args_gen.include_charges:
            #     assert_correctly_masked(charges.float(), node_mask)

            context = context[:, 0]  # [B, 1]
            context = context * self.prop_dist.normalizer[self.target]['mad'] + self.prop_dist.normalizer[self.target]['mean']

            tot_samples.append((one_hot, charges, x, node_mask, context))
        
        return [torch.cat([tot_samples[batch_id][_] for batch_id in range(len(tot_samples))], dim=0)
                for _ in range(len(tot_samples[0]))]

    @staticmethod
    def tensor_to_obj(x: Union[torch.Tensor, List[torch.Tensor]]):
        """
            Given a tensor represneting a batch of molecules, convert it to a list of molecule objects
        """
        one_hot, charges, x, node_mask, target = x  # [B, N, 5], [B, N, 0], [B, N, 3], [B, N], [B, 1]
        one_hot = one_hot.detach().cpu().numpy()
        charges = charges.detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        node_mask = node_mask.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        # list of (one_hot, charges, x, node_mask), for each molecule
        return [(one_hot[_], charges[_], x[_], node_mask[_], target[_]) for _ in range(one_hot.shape[0])]

    @staticmethod
    def obj_to_tensor(objs: List[Any]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
            convert a list of molecule objects into torch tensors that are used for sampling
        """
        batched_one_hot = np.stack([objs[_][0] for _ in range(len(objs))], axis=0)
        batched_one_hot = torch.from_numpy(batched_one_hot)
        batched_charges = np.stack([objs[_][1] for _ in range(len(objs))], axis=0)
        batched_charges = torch.from_numpy(batched_charges)
        batched_x = np.stack([objs[_][2] for _ in range(len(objs))], axis=0)
        batched_x = torch.from_numpy(batched_x)
        batched_node_mask = np.stack([objs[_][3] for _ in range(len(objs))], axis=0)
        batched_node_mask = torch.from_numpy(batched_node_mask)
        batched_target = np.stack([objs[_][4] for _ in range(len(objs))], axis=0)
        batched_target = torch.from_numpy(batched_target)

        return [batched_one_hot, batched_charges, batched_x, batched_node_mask, batched_target]