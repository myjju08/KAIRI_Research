from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
import torch as th
import enum
import math
from diffusion.transformer.openai import ModelMeanType, ModelVarType, LossType

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

        self.vae = None
        self.model_mean_type = self.diffusion_obj.model_mean_type
        self.model_var_type = self.diffusion_obj.model_var_type
        self.loss_type = self.diffusion_obj.loss_type
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
        **kwargs,
    ) -> th.Tensor:

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
        z0_decoded: th.Tensor,  # Decoded image
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