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

def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts

def duplicate(tensor, *size):
    return tensor.unsqueeze(dim=0).expand(*size, *tensor.shape)

class UViT_VPSDE:
    def __init__(self, beta_min=0.1, beta_max=20):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
    def drift(self, x, t):
        return -0.5 * stp(self.squared_diffusion(t), x)
    def diffusion(self, t):
        return self.squared_diffusion(t) ** 0.5
    def squared_diffusion(self, t):
        return self.beta_0 + t * (self.beta_1 - self.beta_0)
    def squared_diffusion_integral(self, s, t):
        return self.beta_0 * (t - s) + (self.beta_1 - self.beta_0) * (t ** 2 - s ** 2) * 0.5
    def skip_beta(self, s, t):
        return 1. - self.skip_alpha(s, t)
    def skip_alpha(self, s, t):
        x = -self.squared_diffusion_integral(s, t)
        return x.exp()
    def cum_beta(self, t):
        return self.skip_beta(0, t)
    def cum_alpha(self, t):
        return self.skip_alpha(0, t)
    def nsr(self, t):
        return self.squared_diffusion_integral(0, t).expm1()
    
    def snr(self, t):
        return 1. / self.nsr(t)

class UViT_ScoreModel:
    def __init__(self, nnet: torch.nn.Module, pred: str, sde: UViT_VPSDE, T=1):
        assert T == 1
        self.nnet = nnet
        self.pred = pred
        self.sde = sde
        self.T = T
    def predict(self, xt, t, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.to(xt.device)
        if t.dim() == 0:
            t = duplicate(t, xt.size(0))
        
        # 공식 U-ViT와 동일하게 t * 999 사용
        t_for_model = t * 999  # 0~999 범위 (공식과 동일)
        t_float = t.float()
        t_for_model_float = t_for_model.float()
        print(f"[DEBUG] predict input t stats: min={t_float.min().item():.6f}, max={t_float.max().item():.6f}, mean={t_float.mean().item():.6f}")
        print(f"[DEBUG] predict t_for_model stats: min={t_for_model_float.min().item():.6f}, max={t_for_model_float.max().item():.6f}, mean={t_for_model_float.mean().item():.6f}")
        
        return self.nnet(xt, t_for_model, **kwargs)
    def noise_pred(self, xt, t, **kwargs):
        pred = self.predict(xt, t, **kwargs)
        
        # 디버깅: noise_pred 계산 확인
        print(f"[DEBUG] noise_pred - pred norm: {pred.norm():.6f}")
        print(f"[DEBUG] noise_pred - pred stats: min={pred.min().item():.6f}, max={pred.max().item():.6f}, mean={pred.mean().item():.6f}")
        
        if self.pred == 'noise_pred':
            noise_pred = pred
        elif self.pred == 'x0_pred':
            noise_pred = - stp(self.sde.snr(t).sqrt(), pred) + stp(self.sde.cum_beta(t).rsqrt(), xt)
        else:
            raise NotImplementedError
            
        print(f"[DEBUG] noise_pred - final noise_pred norm: {noise_pred.norm():.6f}")
        return noise_pred
    def x0_pred(self, xt, t, **kwargs):
        # t를 그대로 사용 (근사하지 않음)
        t_float = t.float()
        print(f"[DEBUG] x0_pred input t stats: min={t_float.min().item():.6f}, max={t_float.max().item():.6f}, mean={t_float.mean().item():.6f}")
        print(f"[DEBUG] x0_pred input t shape: {t.shape}, dtype: {t.dtype}")
        
        # xt 값 확인
        print(f"[DEBUG] xt stats: min={xt.min().item():.6f}, max={xt.max().item():.6f}, mean={xt.mean().item():.6f}, std={xt.std().item():.6f}")
        print(f"[DEBUG] xt norm: {torch.norm(xt).item():.6f}")
        print(f"[DEBUG] xt shape: {xt.shape}")
        print(f"[DEBUG] xt numel: {xt.numel()}")
        print(f"[DEBUG] xt L2 norm per element: {torch.norm(xt).item() / xt.numel():.6f}")
        
        pred = self.predict(xt, t, **kwargs)
        print(f"[DEBUG] predict output stats: min={pred.min().item():.6f}, max={pred.max().item():.6f}, mean={pred.mean().item():.6f}")
        
        if self.pred == 'noise_pred':
            cum_alpha = self.sde.cum_alpha(t)
            nsr = self.sde.nsr(t)
            print(f"[DEBUG] cum_alpha stats: min={cum_alpha.min().item():.6f}, max={cum_alpha.max().item():.6f}")
            print(f"[DEBUG] nsr stats: min={nsr.min().item():.6f}, max={nsr.max().item():.6f}")
            
            # VPSDE 계산 확인
            beta_0 = self.sde.beta_0
            beta_1 = self.sde.beta_1
            squared_diffusion_integral = self.sde.squared_diffusion_integral(0, t)
            print(f"[DEBUG] VPSDE params - beta_0: {beta_0}, beta_1: {beta_1}")
            print(f"[DEBUG] squared_diffusion_integral stats: min={squared_diffusion_integral.min().item():.6f}, max={squared_diffusion_integral.max().item():.6f}, mean={squared_diffusion_integral.mean().item():.6f}")
            print(f"[DEBUG] exp(-integral) stats: min={torch.exp(-squared_diffusion_integral).min().item():.6f}, max={torch.exp(-squared_diffusion_integral).max().item():.6f}, mean={torch.exp(-squared_diffusion_integral).mean().item():.6f}")
            
            # U-ViT 공식 코드와 동일한 x0_pred 공식 사용
            x0_pred = stp(cum_alpha.rsqrt(), xt) - stp(nsr.sqrt(), pred)
            print(f"[DEBUG] x0_pred final stats: min={x0_pred.min().item():.6f}, max={x0_pred.max().item():.6f}, mean={x0_pred.mean().item():.6f}")
        elif self.pred == 'x0_pred':
            x0_pred = pred
        else:
            raise NotImplementedError
        return x0_pred
    def score(self, xt, t, **kwargs):
        cum_beta = self.sde.cum_beta(t)
        noise_pred = self.noise_pred(xt, t, **kwargs)
        return stp(-cum_beta.rsqrt(), noise_pred)

class UViT_ReverseSDE:
    def __init__(self, score_model):
        self.sde = score_model.sde
        self.score_model = score_model
    def drift(self, x, t, **kwargs):
        drift = self.sde.drift(x, t)
        diffusion = self.sde.diffusion(t)
        score = self.score_model.score(x, t, **kwargs)
        
        # 디버깅: drift 계산 확인
        print(f"[DEBUG] ReverseSDE drift - sde.drift norm: {drift.norm():.6f}")
        print(f"[DEBUG] ReverseSDE drift - diffusion: {diffusion:.6f}")
        print(f"[DEBUG] ReverseSDE drift - score norm: {score.norm():.6f}")
        print(f"[DEBUG] ReverseSDE drift - diffusion^2 * score norm: {stp(diffusion ** 2, score).norm():.6f}")
        
        return drift - stp(diffusion ** 2, score)
    def diffusion(self, t):
        return self.sde.diffusion(t)

# 더미 UViT 모델 함수 제거 - 실제 UViT 모델을 사용하므로 불필요
# @torch.no_grad()
# def uvit_step(x, t, model, model_name_or_path, image_size, **kwargs):
#     # 이 함수는 더 이상 사용하지 않음 - guide_step에서 실제 UViT 모델을 로드함
#     pass

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
        ts: th.LongTensor,
        alpha_prod_ts: th.Tensor,
        alpha_prod_t_prevs: th.Tensor,
        eta: float,
        class_labels=None,
        cfg_scale=4.0,
        diffusion=None,
        model_type=None,
        model_name_or_path=None,
        image_size=None,
        guidance_scale=100.0,  # 하이퍼파라미터로 노출 (DPS guidance 활성화, 더 강한 강도)
        **kwargs,
    ) -> th.Tensor:
        if model_type == 'uvit':
            # U-ViT DPS 방식 구현 (x_t -> x_0 -> classifier -> grad)
            sde_obj = UViT_VPSDE(beta_min=0.1, beta_max=20)
            
            # U-ViT 모델을 한 번만 로드하도록 캐싱
            if not hasattr(self, '_uvit_score_model'):
                if model_name_or_path is not None:
                    print(f"[DEBUG] Loading UViT model from: {model_name_or_path}")
                    # UViT 모델 클래스 import
                    import sys
                    import os
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'diffusion', 'transformer'))
                    from uvit_cifar10 import UViT
                    
                    # UViT 모델 인스턴스 생성 (CIFAR-10용 설정)
                    uvit_model = UViT(
                        img_size=32,  # CIFAR-10 이미지 크기
                        patch_size=2,  # CIFAR-10용 패치 크기
                        in_chans=3,
                        embed_dim=512,
                        depth=12,
                        num_heads=8,
                        mlp_ratio=4.,
                        num_classes=-1,  # noise prediction
                        use_checkpoint=False,
                        conv=True,
                        skip=True
                    ).to(x.device)
                    
                    # 모델 가중치 로드
                    state_dict = torch.load(model_name_or_path, map_location=x.device)
                    uvit_model.load_state_dict(state_dict, strict=False)
                    # UViT 모델을 train 모드로 설정하여 gradient 계산 보장
                    uvit_model.train()
                    
                    print(f"[DEBUG] UViT model loaded successfully")
                    self._uvit_score_model = UViT_ScoreModel(uvit_model, pred='noise_pred', sde=sde_obj)
                else:
                    # fallback: 전달받은 model 사용
                    print(f"[DEBUG] Using provided model for UViT")
                    self._uvit_score_model = UViT_ScoreModel(model, pred='noise_pred', sde=sde_obj)
            
            score_model = self._uvit_score_model
                
            rsde = UViT_ReverseSDE(score_model)
            sample_steps = kwargs.get('sample_steps', 50)
            eps = 1e-3
            T = 1
            # 공식 U-ViT와 동일하게 timesteps 생성
            timesteps = np.append(0., np.linspace(eps, T, sample_steps))
            timesteps = torch.tensor(timesteps).to(x)
            print(f"[DEBUG] timesteps: {timesteps[:5]} ... {timesteps[-5:]}")  # 디버깅용
            
            # 공식 U-ViT와 동일하게 timestep 순서 수정
            # t_idx는 49, 48, 47, ..., 0 순서 (reversed)
            # timesteps는 [0.0, 0.02, 0.04, ..., 1.0] 순서
            # 공식 U-ViT: for s, t in zip(timesteps, timesteps[1:])[::-1]
            t_idx = t[0].item() if isinstance(t, torch.Tensor) else int(t)
            
            # 공식 U-ViT와 동일하게 s, t 순서로 매핑
            if t_idx >= len(timesteps) - 1:
                print(f"[WARNING] t_idx={t_idx} >= {len(timesteps)-1}, skipping step")
                return x
                
            s = timesteps[t_idx]      # s: 현재 timestep
            t_val = timesteps[t_idx+1]  # t: 다음 timestep (s > t이므로 양수)
            
            if t_val >= 1.0:
                print(f"[WARNING] t_val={t_val:.4f} >= 1.0, skipping step")
                return x
            target_class = class_labels[0].item() if class_labels is not None else 6  # 기본값을 6으로 변경

            # U-ViT는 0~1 범위의 연속적인 timestep을 사용해야 함
            # 로그를 보면 t가 0~50 범위로 들어오므로 0~1로 정규화
            t_cont = t.float() / 50.0  # 0~1 범위로 정규화 (50 steps 가정)
            t_float = t.float()
            t_cont_float = t_cont.float()
            print(f"[DEBUG] guide_step original t stats: min={t_float.min().item():.6f}, max={t_float.max().item():.6f}, mean={t_float.mean().item():.6f}")
            print(f"[DEBUG] guide_step t_cont stats: min={t_cont_float.min().item():.6f}, max={t_cont_float.max().item():.6f}, mean={t_cont_float.mean().item():.6f}")
            
            # U-ViT 공식 코드와 동일하게 SDE 기반 sampling 사용
            from diffusion.transformer.uvit_sde import euler_maruyama, ReverseSDE
            
            # dt = s - t_val (s > t_val이므로 양수)
            
            # x_t에 grad 연결
            x_with_grad = x.detach().clone().requires_grad_(True)
            
            # 1. x_0 복원 (chain rule 적용)
            x0_pred = score_model.x0_pred(x_with_grad, t_cont)
            if torch.isnan(x0_pred).any() or torch.isinf(x0_pred).any():
                print(f'[ERROR] x0_pred contains NaN/Inf! min={x0_pred.min().item()}, max={x0_pred.max().item()}')
            else:
                print(f'[DEBUG] x0_pred stats: min={x0_pred.min().item()}, max={x0_pred.max().item()}, mean={x0_pred.mean().item()}, std={x0_pred.std().item()}')
            
            # 2. Gradient 계산 (guidance_scale이 0이면 zero gradient 사용)
            if guidance_scale == 0.0:
                print(f"[DEBUG] guidance_scale=0.0, using zero gradient")
                grad_xt = torch.zeros_like(x)
                step_log = f"[STEP {t_idx}] t_val: {t_val:.4f}, x norm: {x.norm():.4f}, grad_xt norm: {grad_xt.norm():.4f} (ZERO)"
            else:
                # x_0에 classifier 적용 및 log_prob 계산, chain rule로 x_t에 대한 gradient 계산
            grad_xt = self._compute_gradient_wrt_x0_uvit(x_with_grad, target_class, t_cont, score_model=score_model)
            if grad_xt is None:
                grad_xt = torch.zeros_like(x)
            if torch.isnan(grad_xt).any() or torch.isinf(grad_xt).any():
                print(f"[WARNING] grad_xt contains NaN/Inf, using zero gradient")
                grad_xt = torch.zeros_like(x)
            
            # Gradient 방향 확인을 위해 clipping 제거
            grad_norm = grad_xt.norm()
            print(f"[DEBUG] Original gradient norm: {grad_norm:.4f}")
            print(f"[DEBUG] Gradient direction check - grad_xt mean: {grad_xt.mean():.4f}, std: {grad_xt.std():.4f}")
            # Gradient clipping 제거 - 원래 gradient 방향 유지
                
            # 스텝별 class 6 확률 추적 및 gradient norm 확인
            step_log = f"[STEP {t_idx}] t_val: {t_val:.4f}, x norm: {x.norm():.4f}, grad_xt norm: {grad_xt.norm():.4f}"
            print(step_log)
            
            # 로그 파일에 저장
            import os
            log_dir = os.path.dirname(self.args.logging_dir) if hasattr(self, 'args') else 'logs'
            step_log_file = os.path.join(log_dir, 'step_logs.txt')
            with open(step_log_file, 'a') as f:
                f.write(step_log + '\n')
            
            # 3. Classifier 디버깅 (guidance_scale이 0이면 건너뛰기)
            if guidance_scale != 0.0:
            # Classifier 출력 디버깅 (x_0 기준) - 스텝별 확률 변화 추적
            with torch.no_grad():
                x0_img = (x0_pred + 1) / 2 # No clamp on x0_img here
                # CIFAR-10 classifier에는 공식 mean/std 정규화만 적용
                import torchvision.transforms as T
                normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
                x0_img_norm = torch.stack([normalize(img) for img in x0_img])
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
                            frog_probs = all_probs[:, 6]  # class 6 (개구리)
                            
                            # 스텝별 상세 로깅
                            class6_log = f"[STEP {t_idx}] Class 6 (frog) probabilities:"
                            class6_log += f"\n  - Min: {frog_probs.min():.4f}, Max: {frog_probs.max():.4f}, Mean: {frog_probs.mean():.4f}"
                            class6_log += f"\n  - Std: {frog_probs.std():.4f}, Median: {frog_probs.median():.4f}"
                            
                            # 상위 5개 샘플의 class 6 확률
                            top5_indices = torch.topk(frog_probs, 5).indices
                            top5_probs = frog_probs[top5_indices]
                            class6_log += f"\n  - Top 5 samples class 6 probs: {top5_probs.cpu().numpy()}"
                            
                            # 전체 클래스 분포 (첫 번째 샘플 기준)
                            first_sample_probs = all_probs[0].cpu().numpy()
                            class6_log += f"\n  - First sample all class probs: {first_sample_probs}"
                            
                            # Class 6이 가장 높은 확률을 가진 샘플 수
                            max_class_indices = torch.argmax(all_probs, dim=1)
                            class6_count = (max_class_indices == 6).sum().item()
                            total_samples = all_probs.shape[0]
                            class6_log += f"\n  - Samples with class 6 as max: {class6_count}/{total_samples} ({class6_count/total_samples*100:.1f}%)"
                            
                            print(class6_log)
                            
                            # 로그 파일에 저장
                            with open(step_log_file, 'a') as f:
                                f.write(class6_log + '\n')
                            
                        else:
                            classifier_output = classifier(x0_img_norm)
                            print(f"[STEP {t_idx}] Raw classifier output: {classifier_output[:5]}")
                            print(f"[STEP {t_idx}] Classifier output shape: {classifier_output.shape}")
                else:
                    print(f"[STEP {t_idx}] Classifier not found, skipping classifier check")
            else:
                print(f"[DEBUG] guidance_scale=0.0, skipping classifier computation")
            
            # 4. SDE Step (U-ViT 공식과 동일)
            with torch.no_grad():
                # SDE drift와 diffusion 계산 (공식 U-ViT와 동일)
                rsde = ReverseSDE(score_model)
                drift = rsde.drift(x, t_val)  # 현재 timestep t_val 사용
                diffusion = rsde.diffusion(t_val)  # 현재 timestep t_val 사용
                dt = s - t_val  # s > t_val이므로 양수
                
                print(f"[DEBUG] SDE step - drift norm: {drift.norm():.6f}, diffusion: {diffusion:.6f}, dt: {dt:.6f}")
                print(f"[DEBUG] SDE step - x norm: {x.norm():.6f}, drift*dt norm: {(drift*dt).norm():.6f}")
                
                # 기본 SDE step (공식 U-ViT와 동일)
                mean = x + drift * dt
                
                # DPS guidance 추가 (guidance_scale이 0이면 추가하지 않음)
                if guidance_scale != 0.0:
                    guidance_term = guidance_scale * (diffusion**2) * grad_xt
                    mean = mean + guidance_term
                    print(f"[DEBUG] Guidance term norm: {guidance_term.norm():.4f}, guidance_scale: {guidance_scale}")
                    print(f"[DEBUG] Mean before guidance: {(x + drift * dt).norm():.4f}, after guidance: {mean.norm():.4f}")
                    guidance_strength = guidance_term.norm() / (x + drift * dt).norm()
                    print(f"[DEBUG] Guidance strength: {guidance_strength:.4f}")
                    print(f"[DEBUG] Guidance term magnitude: {guidance_term.norm():.4f}")
                    print(f"[DEBUG] Base mean magnitude: {(x + drift * dt).norm():.4f}")
                    print(f"[DEBUG] Relative guidance strength: {guidance_strength*100:.2f}%")
                else:
                    print(f"[DEBUG] guidance_scale=0.0, no guidance applied")
                
                sigma = diffusion * (-dt).sqrt()
                
                # 최종 step (공식 U-ViT와 동일) - stp 함수 사용
                if s != 0:  # U-ViT 공식과 동일한 조건
                    x_next = mean + stp(sigma, torch.randn_like(x))
                else:
                    x_next = mean
                
                print(f"[DEBUG] Final step - mean norm: {mean.norm():.4f}, x_next norm: {x_next.norm():.4f}")
                
                # NaN/Inf 체크
                if torch.isnan(x_next).any() or torch.isinf(x_next).any():
                    print(f"[ERROR] x_next contains NaN/Inf, using mean only")
                    x_next = mean
                
                # 메모리 정리
                torch.cuda.empty_cache()
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
            # 1. x_t에 requires_grad 설정 - 중요한 수정: 원본 x_t를 직접 수정하지 않고 새로 생성
            x_t_with_grad = x_t.detach().clone().requires_grad_(True)
            
            # 2. t를 0~1 범위로 정규화
            t_cont = t.float() / 50.0  # 0~1 범위로 정규화 (50 steps 가정)
            
            # t=1에서도 gradient 계산 - 조건 제거
            # if t_cont.max() > 0.99:  # STEP 1 근처 (더 관대한 조건)
            #     print(f"[DEBUG] STEP 1 detected (t={t_cont.max():.4f}), using zero gradient")
            #     return th.zeros_like(x_t)
                
            # 디버깅: t 값 확인 - dtype 변환 추가
            t_cont_float = t_cont.float()  # Long을 Float로 변환
            print(f"[DEBUG] _compute_gradient_wrt_x0_uvit input t stats: min={t.min().item():.6f}, max={t.max().item():.6f}, mean={t.mean().item():.6f}")
            print(f"[DEBUG] _compute_gradient_wrt_x0_uvit t_cont stats: min={t_cont_float.min().item():.6f}, max={t_cont_float.max().item():.6f}, mean={t_cont_float.mean().item():.6f}")
            
            # 3. x_0 복원 (score_model 필요)
            if score_model is None:
                raise ValueError('score_model must be provided for x_t -> x_0 변환')
            
            # UViT 모델 상태 변경하지 않음 - gradient 계산만 위해 requires_grad 설정
            if hasattr(score_model, 'nnet'):
                # 모델 상태는 변경하지 않고 gradient 계산만 활성화
                pass
                
            # U-ViT 공식 코드와 동일한 SDE 기반 x0_pred 계산
            # x0_pred = stp(self.sde.cum_alpha(t).rsqrt(), xt) - stp(self.sde.nsr(t).sqrt(), pred)
            x0_pred = score_model.x0_pred(x_t_with_grad, t_cont)
            
            # 디버깅: x0_pred 값 확인
            print(f"[DEBUG] x0_pred stats: min={x0_pred.min().item():.4f}, max={x0_pred.max().item():.4f}, mean={x0_pred.mean().item():.4f}")
            print(f"[DEBUG] x0_pred requires_grad: {x0_pred.requires_grad}")
            
            # 4. 이미지 변환 및 정규화 - U-ViT 공식 코드와 동일하게 처리
            x0_img = (x0_pred + 1) / 2
            # x0_img = x0_img.clamp(0, 1)  # gradient flow를 위해 clamp 제거
            
            # x0_pred와 classifier 입력값 디버깅
            print(f"[DEBUG] x0_pred stats - min: {x0_pred.min():.4f}, max: {x0_pred.max():.4f}, mean: {x0_pred.mean():.4f}")
            print(f"[DEBUG] x0_img stats - min: {x0_img.min():.4f}, max: {x0_img.max():.4f}, mean: {x0_img.mean():.4f}")
            print(f"[DEBUG] x0_img norm: {x0_img.norm():.4f}")
            
            # CIFAR-10 정규화 (tensor 연산만 사용)
            mean = th.tensor([0.4914, 0.4822, 0.4465], device=x0_img.device, dtype=x0_img.dtype)[None, :, None, None]
            std = th.tensor([0.2471, 0.2435, 0.2616], device=x0_img.device, dtype=x0_img.dtype)[None, :, None, None]
            x0_img_norm = (x0_img - mean) / std
            
            print(f"[DEBUG] x0_img_norm stats - min: {x0_img_norm.min():.4f}, max: {x0_img_norm.max():.4f}, mean: {x0_img_norm.mean():.4f}")
            print(f"[DEBUG] x0_img_norm requires_grad: {x0_img_norm.requires_grad}")
            
            # 5. Classifier 찾기
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
            
            # Classifier 상태 변경하지 않음 - gradient 계산만 활성화
            # classifier.train()  # 모델 상태 변경하지 않음
            # for param in classifier.parameters():
            #     param.requires_grad = True
                
            # 6. Classifier forward pass
            logits = classifier(x0_img_norm)
            print(f"[DEBUG] logits shape: {logits.shape}, requires_grad: {logits.requires_grad}")
            
            # 7. Log probability 계산
            if logits.ndim == 1:
                selected_log_probs = logits
            else:
                probs = th.nn.functional.softmax(logits, dim=1)
                log_probs = th.log(probs + 1e-8)  # 수치 안정성을 위한 epsilon 추가
                batch_size = x0_img.shape[0]
                if isinstance(target_class, int):
                    selected_log_probs = log_probs[range(batch_size), target_class]
                    print(f"[DEBUG] Using target_class: {target_class} for gradient computation")
                else:
                    selected_log_probs = log_probs[range(batch_size), 6]  # class 6 (frog)에 대한 gradient 계산
                    print(f"[DEBUG] Using default class 6 (frog) for gradient computation")
                    
            print(f"[DEBUG] selected_log_probs shape: {selected_log_probs.shape}, requires_grad: {selected_log_probs.requires_grad}")
            print(f"[DEBUG] selected_log_probs values: {selected_log_probs.detach().cpu().numpy()}")
            
            # 8. Gradient 계산 - NEGATIVE log probability를 최소화 (확률 최대화)
            # 더 강한 guidance를 위해 각 샘플별로 개별적으로 처리
            log_prob_sum = -selected_log_probs.sum()  # NEGATIVE log probability 합계 (확률 최대화)
            print(f"[DEBUG] log_prob_sum (negative): {log_prob_sum.item()}, requires_grad: {log_prob_sum.requires_grad}")
            print(f"[DEBUG] Individual log probs: {selected_log_probs.detach().cpu().numpy()}")
            print(f"[DEBUG] Target class probabilities: {torch.exp(selected_log_probs).detach().cpu().numpy()}")
            
            # gradient 계산 전에 backward() 사용하여 더 안정적인 gradient 계산
            log_prob_sum.backward(retain_graph=True)
            grad_xt = x_t_with_grad.grad.clone() if x_t_with_grad.grad is not None else th.zeros_like(x_t)
            
            # gradient 초기화
            x_t_with_grad.grad.zero_() if x_t_with_grad.grad is not None else None
            
            # 디버깅: gradient norm 확인
            print(f"[DEBUG] grad_xt norm: {grad_xt.norm().item():.4f}, grad_xt std: {grad_xt.std().item():.4f}")
            print(f"[DEBUG] grad_xt min: {grad_xt.min().item():.4f}, grad_xt max: {grad_xt.max().item():.4f}")
            
            # Gradient 방향 분석
            print(f"[DEBUG] Gradient direction analysis:")
            print(f"  - Positive gradients: {(grad_xt > 0).sum().item()}/{grad_xt.numel()} ({(grad_xt > 0).float().mean().item():.2%})")
            print(f"  - Negative gradients: {(grad_xt < 0).sum().item()}/{grad_xt.numel()} ({(grad_xt < 0).float().mean().item():.2%})")
            print(f"  - Zero gradients: {(grad_xt == 0).sum().item()}/{grad_xt.numel()} ({(grad_xt == 0).float().mean().item():.2%})")
            
            # gradient가 0인 경우 추가 디버깅
            if grad_xt.norm() < 1e-8:
                print(f"[WARNING] grad_xt is zero! Checking computation chain...")
                print(f"[DEBUG] x0_pred grad_fn: {x0_pred.grad_fn}")
                print(f"[DEBUG] x0_img grad_fn: {x0_img.grad_fn}")
                print(f"[DEBUG] x0_img_norm grad_fn: {x0_img_norm.grad_fn}")
                print(f"[DEBUG] logits grad_fn: {logits.grad_fn}")
                print(f"[DEBUG] selected_log_probs grad_fn: {selected_log_probs.grad_fn}")
                print(f"[DEBUG] log_prob_sum grad_fn: {log_prob_sum.grad_fn}")
                
                # UViT 모델 상태 확인
                if hasattr(score_model, 'nnet'):
                    print(f"[DEBUG] UViT model training mode: {score_model.nnet.training}")
                    print(f"[DEBUG] UViT model requires_grad: {next(score_model.nnet.parameters()).requires_grad}")
                
                # Classifier 상태 확인
                print(f"[DEBUG] Classifier training mode: {classifier.training}")
                print(f"[DEBUG] Classifier requires_grad: {next(classifier.parameters()).requires_grad}")
                
                # 간단한 테스트: x_t_with_grad에 직접 loss 적용
                test_loss = x_t_with_grad.sum()
                test_loss.backward()
                test_grad = x_t_with_grad.grad.clone()
                print(f"[DEBUG] Test gradient norm: {test_grad.norm().item():.4f}")
                x_t_with_grad.grad.zero_()
                
                # 추가 테스트: x0_pred에 직접 loss 적용
                x0_pred_loss = x0_pred.sum()
                x0_pred_loss.backward()
                x0_pred_grad = x_t_with_grad.grad.clone()
                print(f"[DEBUG] x0_pred gradient norm: {x0_pred_grad.norm().item():.4f}")
                x_t_with_grad.grad.zero_()
            
            # Gradient clipping 완전 제거 - gradient 크기 정보 보존
            # if grad_xt.norm() > 100.0:  # 10.0에서 100.0으로 증가
            #     grad_xt = grad_xt * 100.0 / grad_xt.norm()
            #     print(f"[DEBUG] Gradient clipped to norm: {grad_xt.norm().item():.4f}")
            
            # Gradient scaling - 더 강한 guidance를 위해 gradient 크기 증가
                
            # 메모리 정리
            del x_t_with_grad, x0_pred, x0_img, x0_img_norm, logits, selected_log_probs, log_prob_sum
            th.cuda.empty_cache()
            
            return grad_xt
            
        except Exception as e:
            print(f'U-ViT gradient computation failed: {e}')
            import traceback
            traceback.print_exc()
            th.cuda.empty_cache()
            return th.zeros_like(x_t)

def test_gradient_computation():
    """
    Gradient 계산을 테스트하는 함수
    """
    import torch
    import torch.nn as nn
    
    # 간단한 테스트 모델
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
            
        def forward(self, x, t, y=None):
            return self.conv(x)
    
    # 테스트 데이터
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    t = torch.tensor([0.5])
    model = TestModel()
    
    # UViT 설정
    sde_obj = UViT_VPSDE(beta_min=0.1, beta_max=20)
    score_model = UViT_ScoreModel(model, pred='noise_pred', sde=sde_obj)
    
    # x0_pred 테스트
    x0_pred = score_model.x0_pred(x, t)
    print(f"x0_pred requires_grad: {x0_pred.requires_grad}")
    print(f"x0_pred grad_fn: {x0_pred.grad_fn}")
    
    # 간단한 loss로 gradient 테스트
    loss = x0_pred.sum()
    loss.backward()
    print(f"x.grad norm: {x.grad.norm()}")
    
    return x.grad is not None and x.grad.norm() > 0

if __name__ == "__main__":
    # Gradient 테스트 실행
    success = test_gradient_computation()
    print(f"Gradient test {'PASSED' if success else 'FAILED'}")