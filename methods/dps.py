from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
import torch as th
import enum
import math
from diffusion.transformer.openai import ModelMeanType, ModelVarType, LossType
import torch.nn.functional as F
import torchvision.transforms as T

# === UViT용 SDE/ScoreModel/샘플 step 함수 ===
import torch
import numpy as np

def stp(s, ts: torch.Tensor):
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts

def duplicate(tensor, *size):
    return tensor.unsqueeze(dim=0).expand(*size, *tensor.shape)

# U-ViT 원본 SDE 클래스 사용
from sde import VPSDE as UViT_VPSDE

# U-ViT 원본 클래스들 사용
from sde import ScoreModel as UViT_ScoreModel, ReverseSDE as UViT_ReverseSDE

@torch.no_grad()
def uvit_step(x, t, model, model_name_or_path, image_size, **kwargs):
    # 실제 UViT 모델 사용
    from libs.uvit import UViT
    uvit_model = UViT(
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
    ).to(x.device)
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

class DPSGuidance(BaseGuidance):

    def __init__(self, args, diffusion=None, **kwargs):
        super(DPSGuidance, self).__init__(args, **kwargs)
        
        # diffusion 인스턴스가 넘어오면 그 속성 사용 (UViT 샘플러 지원)
        if diffusion is not None:
            self.diffusion_obj = diffusion
            # UViT 샘플러의 diffusion 스케줄 속성 복사
            for attr in [
                'betas', 'alphas_cumprod', 'alphas_cumprod_prev', 'alphas_cumprod_next',
                'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'log_one_minus_alphas_cumprod',
                'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod',
                'posterior_variance', 'posterior_log_variance_clipped',
                'posterior_mean_coef1', 'posterior_mean_coef2',
                'num_timesteps'
            ]:
                if hasattr(diffusion, attr):
                    setattr(self, attr, getattr(diffusion, attr))
        else:
            self.diffusion_obj = create_diffusion(
                timestep_respacing=str(args.inference_steps),
                noise_schedule="linear",
                use_kl=False,
                sigma_small=False,
                predict_xstart=False,
                learn_sigma=True,
                rescale_learned_sigmas=False,
                diffusion_steps=args.train_steps
            )
            self.betas = self.diffusion_obj.betas
            self.num_timesteps = self.diffusion_obj.num_timesteps
            self.alphas_cumprod = self.diffusion_obj.alphas_cumprod
            self.alphas_cumprod_prev = self.diffusion_obj.alphas_cumprod_prev
            self.alphas_cumprod_next = self.diffusion_obj.alphas_cumprod_next
            self.sqrt_alphas_cumprod = self.diffusion_obj.sqrt_alphas_cumprod
            self.sqrt_one_minus_alphas_cumprod = self.diffusion_obj.sqrt_one_minus_alphas_cumprod
            self.log_one_minus_alphas_cumprod = self.diffusion_obj.log_one_minus_alphas_cumprod
            self.sqrt_recip_alphas_cumprod = self.diffusion_obj.sqrt_recip_alphas_cumprod
            self.sqrt_recipm1_alphas_cumprod = self.diffusion_obj.sqrt_recipm1_alphas_cumprod
            self.posterior_variance = self.diffusion_obj.posterior_variance
            self.posterior_log_variance_clipped = self.diffusion_obj.posterior_log_variance_clipped
            self.posterior_mean_coef1 = self.diffusion_obj.posterior_mean_coef1
            self.posterior_mean_coef2 = self.diffusion_obj.posterior_mean_coef2
        self.vae = None
        self.model_mean_type = getattr(self.diffusion_obj, 'model_mean_type', None)
        self.model_var_type = getattr(self.diffusion_obj, 'model_var_type', None)
        self.loss_type = getattr(self.diffusion_obj, 'loss_type', None)

    def guide_step(
        self,
        x: th.Tensor,
        t: th.Tensor,
        model: th.nn.Module,
        ts: th.LongTensor = None,
        alpha_prod_ts: th.Tensor = None,
        alpha_prod_t_prevs: th.Tensor = None,
        eta: float = 0.0,
        class_labels=None,
        cfg_scale=4.0,
        diffusion=None,
        model_type=None,
        model_name_or_path=None,
        image_size=None,
        **kwargs,
    ) -> th.Tensor:
        if model_type == 'uvit':
            # UViT SDE 기반 DPS 구현
            from sde import VPSDE, ScoreModel, ReverseSDE
            from libs.uvit import UViT
            
            # UViT 모델 로드 (캐싱 가능)
            if not hasattr(self, '_uvit_model') or self._uvit_model is None:
                self._uvit_model = UViT(
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
                    norm_layer=torch.nn.LayerNorm,
                    use_checkpoint=False
                ).to(x.device)
                state_dict = torch.load(model_name_or_path, map_location=x.device)
                self._uvit_model.load_state_dict(state_dict)
                self._uvit_model.eval()
            
            # SDE 설정
            sde_obj = VPSDE(beta_min=0.1, beta_max=20)
            score_model = ScoreModel(self._uvit_model, pred='noise_pred', sde=sde_obj)
            rsde = ReverseSDE(score_model)
            
            # SDE timesteps 설정
            sample_steps = kwargs.get('sample_steps', 50)
            eps = 1e-3
            T = 1
            timesteps = np.linspace(eps, T, sample_steps+1)
            timesteps = torch.tensor(timesteps).to(x)
            
            # 현재 timestep 계산
            t_idx = t[0].item() if isinstance(t, torch.Tensor) else int(t)
            s = timesteps[t_idx]
            t_val = timesteps[t_idx+1] if t_idx+1 < len(timesteps) else 0.0
            
            if t_val >= 1.0:
                print(f"[WARNING] t_val={t_val:.4f} >= 1.0, skipping step")
                return x
            
            target_class = class_labels[0].item() if class_labels is not None else 1000
            
            # DPS: x_t -> x_0 -> classifier -> gradient
            x_with_grad = x.detach().clone().requires_grad_(True)
            x0_pred = score_model.x0_pred(x_with_grad, t_val)
            
            # NaN/Inf 체크
            if torch.isnan(x0_pred).any() or torch.isinf(x0_pred).any():
                print(f'[ERROR] x0_pred contains NaN/Inf! min={x0_pred.min().item()}, max={x0_pred.max().item()}')
                grad_xt = torch.zeros_like(x)
            else:
                print(f'[DEBUG] x0_pred stats: min={x0_pred.min().item()}, max={x0_pred.max().item()}, mean={x0_pred.mean().item()}, std={x0_pred.std().item()}')
                # Classifier gradient 계산
                grad_xt = self._compute_gradient_wrt_x0_uvit(x_with_grad, target_class, t_val, score_model=score_model)
                if grad_xt is None or torch.isnan(grad_xt).any() or torch.isinf(grad_xt).any():
                    print(f"[WARNING] grad_xt contains NaN/Inf, using zero gradient")
                    grad_xt = torch.zeros_like(x)
            
            print(f"[DEBUG] t_val: {t_val:.4f}, x norm: {x.norm():.4f}, grad_xt norm: {grad_xt.norm():.4f}")
            
            # Classifier 확률값 디버깅
            with torch.no_grad():
                # x0_pred를 이미지 형태로 변환
                x0_img = x0_pred.clamp(-1, 1)
                x0_img = (x0_img + 1) / 2  # [-1,1] -> [0,1]
                x0_img = x0_img.clamp(0, 1)
                
                # CIFAR-10 정규화 적용
                import torchvision.transforms as T
                normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
                x0_img_norm = torch.stack([normalize(img) for img in x0_img])
                
                # Classifier 찾기
                classifier = None
                if hasattr(self, 'guider') and hasattr(self.guider, 'get_guidance'):
                    get_guidance = self.guider.get_guidance
                    if hasattr(get_guidance, 'func') and hasattr(get_guidance.func, '__self__'):
                        guider_obj = get_guidance.func.__self__
                        if hasattr(guider_obj, 'classifier'):
                            classifier = guider_obj.classifier
                
                if classifier is not None:
                    with torch.no_grad():
                        if hasattr(classifier, '_forward'):
                            all_logits = classifier._forward(x0_img_norm)
                            all_probs = torch.softmax(all_logits, dim=1)
                            
                            # 타겟 클래스 (개구리=6) 확률
                            target_probs = all_probs[:, target_class]
                            
                            # 전체 클래스 확률 분포
                            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                            target_class_name = class_names[target_class] if target_class < len(class_names) else f'class_{target_class}'
                            
                            print(f"[DEBUG] Classifier Results (t={t_val:.4f}):")
                            print(f"  Target class ({target_class_name}): prob={target_probs.mean().item():.4f} (min={target_probs.min().item():.4f}, max={target_probs.max().item():.4f})")
                            
                            # 모든 샘플의 frog 확률값 출력
                            frog_probs = all_probs[:, target_class]  # frog = class 6
                            print(f"  All samples frog probs: {[f'{prob:.4f}' for prob in frog_probs.cpu().numpy()]}")
                            
                            # Top-3 클래스 확률
                            top3_probs, top3_indices = torch.topk(all_probs.mean(dim=0), 3)
                            print(f"  Top-3 classes: {[f'{class_names[idx]}({prob:.4f})' for idx, prob in zip(top3_indices.cpu().numpy(), top3_probs.cpu().numpy())]}")
                            
                            # 첫 번째 샘플의 상세 확률
                            if x0_img.shape[0] > 0:
                                first_sample_probs = all_probs[0]
                                print(f"  First sample probs: {[f'{name}:{prob:.4f}' for name, prob in zip(class_names, first_sample_probs.cpu().numpy())]}")
                        else:
                            classifier_output = classifier(x0_img_norm)
                            print(f"[DEBUG] Raw classifier output shape: {classifier_output.shape}")
                else:
                    print(f"[DEBUG] Classifier not found")
            
            # SDE step with DPS guidance
            with torch.no_grad():
                drift = rsde.drift(x, t_val)
                diffusion = rsde.diffusion(t_val)
                dt = s - t_val
                mean = x + drift * dt
                sigma = diffusion * (-dt).sqrt()
                
                                # DPS guidance 적용
                guidance_scale = 1.0  # DPS guidance 활성화
                x_next = mean + sigma * torch.randn_like(x) + guidance_scale * (sigma**2) * grad_xt
                
                if torch.isnan(x_next).any() or torch.isinf(x_next).any():
                    print(f"[ERROR] x_next contains NaN/Inf, using mean only")
                    x_next = mean
                
                print(f"[DEBUG] guidance_scale: {guidance_scale}, mean norm: {mean.norm():.4f}, x_next norm: {x_next.norm():.4f}")
                
                return x_next

        elif model_type in ['transformer']:
            # 기존 DiT step 함수 (기존 코드)
            # 타겟 클래스 사용 (class_labels가 None이면 1000 사용)
            if class_labels is not None:
                target_class = class_labels[0].item() if len(class_labels) > 0 else 1000
            else:
                target_class = 1000
            
            # 1. Predict x0 from current x_t using the diffusion model
            out = self.diffusion_obj.p_sample(
                model.forward_with_cfg, 
                x.float(), 
                t, 
                clip_denoised=False, 
                model_kwargs=dict(y=th.tensor([1000], device=x.device, dtype=th.long), cfg_scale=cfg_scale)
            )
            pred_xstart = out["pred_xstart"]
            
            # 2. Decode to image space for guidance
            z0_decoded = self.vae.decode(pred_xstart / 0.18215, return_dict=False)[0]
            
            # 3. Compute gradient w.r.t. z_t (current latent)
            grad_z_t = self._compute_gradient_wrt_z_t(
                x, pred_xstart, z0_decoded, target_class, t, **kwargs
            )
            
            # 4. Apply DPS update rule according to the paper
            eps = self.diffusion_obj._predict_eps_from_xstart(x, t, pred_xstart)
            
            # Get diffusion parameters
            alpha_t = alpha_prod_ts[t]
            alpha_prev = alpha_prod_t_prevs[t]
            
            # DPS 논문의 정확한 수식 구현
            # x_{t-1} = μ(x_t, t) + Σ(x_t, t) * ∇_{x_t} log p(y|x_0)
            
            # Standard DDIM mean prediction
            sqrt_1m_alpha_t = (1 - alpha_t).sqrt().view(-1, 1, 1, 1)
            sqrt_alpha_prev_alpha_t = (alpha_prev / alpha_t).sqrt().view(-1, 1, 1, 1)
            sqrt_1m_alpha_prev_sigma2 = (1 - alpha_prev).sqrt().view(-1, 1, 1, 1)
            
            # DDIM step without noise
            x_prev_mean = (
                sqrt_alpha_prev_alpha_t * (x - sqrt_1m_alpha_t * eps)
                + sqrt_1m_alpha_prev_sigma2 * eps
            )
            
            # DPS guidance term: posterior variance * gradient
            # DPS 논문에서 posterior variance는 β_t * (1 - α_{t-1}) / (1 - α_t)
            try:
                # DDIM에서 posterior variance는 0이므로, 대신 guidance scale을 사용
                # posterior_var = self.diffusion_obj.posterior_variance[t]
                
                # 대신 DPS 논문에서 제안하는 방법 사용
                # posterior variance 대신 guidance scale 사용
                guidance_scale = 0.3
                
                # Gradient normalization for numerical stability
                grad_norm = grad_z_t.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                normalized_grad = grad_z_t / (grad_norm + 1e-8)
                
                # Apply DPS guidance: x_{t-1} = μ + guidance_scale * normalized_grad
                x_prev = x_prev_mean + guidance_scale * normalized_grad
                
            except Exception as e:
                print(f"DPS guidance computation failed: {e}")
                # Fallback: no guidance
                x_prev = x_prev_mean
            
            # Add noise if eta > 0 (for stochastic sampling)
            if eta > 0:
                # Create nonzero_mask exactly like DiT
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                )  # no noise when t == 0
                
                # Generate noise and add it
                noise = th.randn_like(x)
                sigma = eta * (
                    (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
                ) ** (0.5)
                x_prev = x_prev + nonzero_mask * sigma.view(-1, 1, 1, 1) * noise * 0
            
            out["sample"] = x_prev.float()
            return out

    def _compute_gradient_wrt_z_t(
        self, 
        z_t: th.Tensor,  # Current latent z_t
        pred_xstart: th.Tensor,  # Predicted z_0
        z0_decoded: th.Tensor,  # Decoded imageㅜ
        target_class: int,
        t: th.Tensor,
        **kwargs
    ) -> th.Tensor:
        """
        DPS 논문의 정확한 gradient 계산
        ∇_{z_t} log p(c|z_0) = ∇_{z_0} log p(c|z_0) * ∂z_0/∂z_t
        
        DDIM에서: z_0 = (z_t - √(1-α_t) * ε) / √(α_t)
        따라서: ∂z_0/∂z_t = 1/√(α_t)
        """
        try:
            # 1. pred_xstart (latent)에 requires_grad 설정 및 leaf tensor 보장 (항상 한 줄로 강제)
            pred_xstart_new = pred_xstart.to(dtype=th.float32).detach().clone().requires_grad_(True)

            # 2. pred_xstart → z0_decoded (VAE decode)
            z0_decoded_new = self.vae.decode(pred_xstart_new / 0.18215, return_dict=False)[0]

            # 3. classifier, log_prob, loss 계산
            # 전체 구조에 맞게 정확히 classifier 추출
            classifier = None
            if hasattr(self, 'guider') and hasattr(self.guider, 'get_guidance'):
                get_guidance = self.guider.get_guidance
                if hasattr(get_guidance, 'func') and hasattr(get_guidance.func, '__self__'):
                    guider_obj = get_guidance.func.__self__
                    if hasattr(guider_obj, 'classifier'):
                        classifier = guider_obj.classifier
            if classifier is None:
                raise RuntimeError('No classifier found for guidance gradient computation.')

            # 디버깅: classifier가 실제로 어떤 객체인지 출력
            # print(f"[DEBUG] classifier type: {type(classifier)}, repr: {repr(classifier)}")
            # 2. VAE decode (gradient 필요 여부에 따라 preserve_grad)

            # classifier가 nn.Module이면 직접 호출
            logits = classifier(z0_decoded_new)
            # HuggingfaceClassifier/torchvision 모두 log_prob 반환
            if logits.ndim == 1:
                # 이미 log_prob (selected)만 반환하는 구조
                selected_log_probs = logits
            else:
                # logits에서 softmax-log, target 인덱싱
                probs = th.nn.functional.softmax(logits, dim=1)
                log_probs = th.log(probs)
                batch_size = z0_decoded_new.shape[0]
                if isinstance(target_class, int) or isinstance(target_class, str):
                    selected_log_probs = log_probs[range(batch_size), int(target_class)]
                else:
                    selected_log_probs = th.cat([log_probs[range(batch_size), _] for _ in target_class], dim=0)

            # 4. log probability의 합에 대해 autograd.grad로 gradient 계산
            log_prob_sum = selected_log_probs.sum()
            grad_z0 = th.autograd.grad(
                outputs=log_prob_sum,
                inputs=pred_xstart_new,
                retain_graph=False,
                allow_unused=True
            )[0]

            # 5. 불필요한 변수/그래프 해제 및 메모리 정리
            del pred_xstart_new, z0_decoded_new, log_prob_sum
            th.cuda.empty_cache()
            return grad_z0
        except Exception as e:
            print(f'Gradient computation failed: {e}')
            th.cuda.empty_cache()
            # fallback: 입력 pred_xstart와 같은 shape의 0 tensor 반환
            return th.zeros_like(pred_xstart)

    def _compute_gradient_wrt_x0_uvit(
        self,
        x_t: th.Tensor,  # Noisy input x_t
        target_class: int,
        t: th.Tensor,    # timestep
        score_model=None,
        **kwargs
    ) -> th.Tensor:
        """
        x_t -> x_0 -> classifier log prob을 x_t에 대해 직접 미분 (chain rule 없이 autograd로)
        """
        try:
            # 1. x_t에 requires_grad 설정
            x_t = x_t.to(dtype=th.float32).detach().clone().requires_grad_(True)

            # 2. x_0 복원 (score_model 필요)
            if score_model is None:
                raise ValueError('score_model must be provided for x_t -> x_0 변환')
            x0_pred = score_model.x0_pred(x_t, t)
            x0_img = x0_pred.clamp(-1, 1)
            x0_img = (x0_img + 1) / 2
            x0_img = x0_img.clamp(0, 1)

            # 2-1. CIFAR-10 정규화 (gradient가 끊기지 않게 tensor 연산으로 적용)
            mean = th.tensor([0.4914, 0.4822, 0.4465], device=x0_img.device, dtype=x0_img.dtype)[None, :, None, None]
            std = th.tensor([0.2471, 0.2435, 0.2616], device=x0_img.device, dtype=x0_img.dtype)[None, :, None, None]
            x0_img_norm = (x0_img - mean) / std

            # 3. Classifier 찾기
            classifier = None
            if hasattr(self, 'guider') and hasattr(self.guider, 'get_guidance'):
                get_guidance = self.guider.get_guidance
                if hasattr(get_guidance, 'func') and hasattr(get_guidance.func, '__self__'):
                    guider_obj = get_guidance.func.__self__
                    if hasattr(guider_obj, 'classifier'):
                        classifier = guider_obj.classifier
            if classifier is None:
                print("Warning: No classifier found, using zero gradient")
                return th.zeros_like(x_t)

            logits = classifier(x0_img_norm)
            if logits.ndim == 1:
                selected_log_probs = logits
            else:
                probs = th.nn.functional.softmax(logits, dim=1)
                log_probs = th.log(probs)
                batch_size = x0_img.shape[0]
                if isinstance(target_class, int):
                    selected_log_probs = log_probs[range(batch_size), target_class]
                else:
                    selected_log_probs = log_probs[range(batch_size), 0]

            log_prob_sum = selected_log_probs.sum()
            grad_result = th.autograd.grad(
                outputs=log_prob_sum,
                inputs=x_t,
                retain_graph=False,
                allow_unused=True
            )
            grad_xt = grad_result[0] if grad_result[0] is not None else th.zeros_like(x_t)
            th.cuda.empty_cache()
            return grad_xt
        except Exception as e:
            print(f'U-ViT gradient computation failed: {e}')
            th.cuda.empty_cache()
            return th.zeros_like(x_t)