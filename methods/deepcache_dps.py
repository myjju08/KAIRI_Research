import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .dps import DPSGuidance
from models.deepcache_unet import create_deepcache_unet
import time


class DeepCacheDPSGuidance(DPSGuidance):
    """
    DeepCache를 적용한 DPS Guidance
    x_t → x_0 예측 부분만 DeepCache로 최적화
    """
    
    def __init__(self, args, diffusion=None, **kwargs):
        super().__init__(args, diffusion, **kwargs)
        
        # DeepCache 설정
        self.use_deepcache = getattr(args, 'use_deepcache', False)
        self.cache_interval = getattr(args, 'cache_interval', 1)
        self.cache_layer_id = getattr(args, 'cache_layer_id', None)
        self.cache_block_id = getattr(args, 'cache_block_id', None)
        
        # model_type 설정 (부모 클래스에서 상속)
        self.model_type = getattr(args, 'model_type', 'unet')
        
        # DeepCache UNet 초기화 (나중에 모델 로드 시 적용)
        self.deepcache_unet = None
        self.original_model = None  # 원본 모델 저장용
        
        # 캐시 관련 변수들
        self.prv_features = None
        
        print(f"[DeepCache] DeepCacheDPSGuidance 초기화: use_deepcache={self.use_deepcache}, cache_interval={self.cache_interval}")
    
    def _setup_deepcache_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """DeepCache 모델 설정"""
        if self.deepcache_unet is None:
            self.deepcache_unet = create_deepcache_unet(
                original_unet=model,
                cache_interval=self.cache_interval,
                cache_layer_id=getattr(self.args, 'cache_layer_id', None),
                cache_block_id=getattr(self.args, 'cache_block_id', None),
                clean_step=getattr(self.args, 'clean_step', None)
            )
        return self.deepcache_unet
    
    def guide_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        model: torch.nn.Module,
        ts: torch.LongTensor = None,
        alpha_prod_ts: torch.Tensor = None,
        alpha_prod_t_prevs: torch.Tensor = None,
        eta: float = 0.0,
        class_labels=None,
        cfg_scale=4.0,
        diffusion=None,
        model_type=None,
        model_name_or_path=None,
        image_size=None,
        guidance_scale=None,
        start_gradient=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        DeepCache 적용된 guide_step
        Diffusion step은 원본 모델 사용, x_0 예측 부분만 DeepCache 적용
        """
        
        # DeepCache를 사용하지 않을 때는 부모 클래스의 guide_step 호출
        if not self.use_deepcache:
            return super().guide_step(
                x, t, model, ts, alpha_prod_ts, alpha_prod_t_prevs, eta,
                class_labels, cfg_scale, diffusion, self.model_type, model_name_or_path,
                image_size, guidance_scale, start_gradient, **kwargs
            )
        
        # UNet 모델 타입일 때는 DeepCache 로직 사용
        if self.model_type == 'unet':
            return self._guide_step_unet(x, t, model, ts, alpha_prod_ts, alpha_prod_t_prevs, eta, **kwargs)
        else:
            # UViT나 Transformer는 부모 클래스의 guide_step 호출
            return super().guide_step(
                x, t, model, ts, alpha_prod_ts, alpha_prod_t_prevs, eta,
                class_labels, cfg_scale, diffusion, self.model_type, model_name_or_path,
                image_size, guidance_scale, start_gradient, **kwargs
            )
    
    def _guide_step_unet(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        model: torch.nn.Module,
        ts: torch.LongTensor = None,
        alpha_prod_ts: torch.Tensor = None,
        alpha_prod_t_prevs: torch.Tensor = None,
        eta: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """UNet용 DeepCache guide_step - x0 예측에만 DeepCache 적용"""
        
        # 현재 timestep 확인 (정규화된 값 사용)
        if ts is not None:
            # t는 ts 배열의 인덱스이므로, 정규화된 값은 50 - t
            if isinstance(t, int):
                current_t = 50 - t
            elif isinstance(t, torch.Tensor):
                current_t = 50 - t.item()
            else:
                current_t = 50 - int(t)
        else:
            # ts가 없으면 t 값 사용
            if isinstance(t, int):
                current_t = t
            elif isinstance(t, torch.Tensor):
                if t.dim() == 0:
                    current_t = t.item()
                elif t.dim() == 1:
                    current_t = t[0].item()
                else:
                    current_t = t.item()
            else:
                current_t = int(t)
        
        # 1. epsilon 예측: 원본 UNet 사용 (DeepCache 사용 안함)
        alpha_prod_t = alpha_prod_ts[t]
        alpha_prod_t_prev = alpha_prod_t_prevs[t]
        t_tensor = ts[t] if ts is not None else t
        
        # 원본 UNet으로 epsilon 예측
        epsilon = model(x, t_tensor)
        x_prev = self._predict_x_prev_from_eps(x, epsilon, alpha_prod_t, alpha_prod_t_prev, eta, t_tensor, **kwargs)
        
        # 2. x0 예측 (classifier gradient 계산): DeepCache 사용
        # DeepCache 모델 설정
        deepcache_model = self._setup_deepcache_model(model)
        
        # DeepCache 모델의 캐시 업데이트 스텝 확인
        is_cache_update_step = self.deepcache_unet._is_cache_update_step(current_t)
        
        if is_cache_update_step:
            print(f"[DeepCache] 캐시 업데이트 스텝 (t={current_t}): x0 예측에서 전체 계산")
        else:
            print(f"[DeepCache] 캐싱 스텝 (t={current_t}): x0 예측에서 캐시 사용")
        
        # DeepCache를 사용한 x0 예측 함수
        def func(zt):
            # DeepCache 모델로 x0 예측
            x0 = self._deepcache_x0_predict(zt, deepcache_model, t_tensor, alpha_prod_t)
            return x0
        
        # gradient 디버깅: guidance 계산 전 입력값 확인
        x_for_grad = x.clone().detach().requires_grad_()
        print(f"[Gradient Debug] t={t}, x_for_grad.requires_grad: {x_for_grad.requires_grad}")
        
        guidance = self.guider.get_guidance(
            x_for_grad, 
            func, 
            **kwargs
        )
        
        # gradient 디버깅: guidance 결과 확인
        if guidance is not None:
            grad_norm = torch.norm(guidance).item()
            grad_max = torch.max(torch.abs(guidance)).item()
            print(f"[Gradient Debug] t={t}, grad_norm: {grad_norm:.6f}, grad_max: {grad_max:.6f}")
            
            # gradient가 거의 0인지 확인
            if grad_norm < 1e-8:
                print(f"[Gradient Warning] t={t}, Gradient is almost zero! This might be a caching step.")
        else:
            print(f"[Gradient Debug] t={t}, guidance is None")
        
        # follow the schedule of DPS paper
        logp_norm = self.guider.get_guidance(x.clone().detach(), func, return_logp=True, check_grad=False, **kwargs)
        
        x_prev = x_prev + self.args.guidance_strength * guidance / torch.abs(logp_norm.view(-1, * ([1] * (len(x_prev.shape) - 1)) ))
            
        return x_prev
    
    def _deepcache_x0_predict(self, zt, deepcache_model, t, alpha_prod_t):
        """DeepCache를 사용한 x0 예측"""
        # timesteps를 올바른 형태로 변환
        if isinstance(t, int):
            timesteps_tensor = torch.tensor([t], device=zt.device, dtype=torch.long)
        elif isinstance(t, torch.Tensor):
            if t.dim() == 0:
                timesteps_tensor = t.unsqueeze(0).to(zt.device)
            else:
                timesteps_tensor = t.to(zt.device)
        else:
            timesteps_tensor = torch.tensor([t], device=zt.device, dtype=torch.long)
        
        # gradient 디버깅: epsilon 예측 전 입력값 확인
        print(f"[Epsilon Debug] zt.requires_grad: {zt.requires_grad}, zt.grad_fn: {zt.grad_fn}")
        
        # DeepCache UNet으로 epsilon 예측 (내부적으로 캐싱 적용)
        epsilon = deepcache_model(zt, timesteps_tensor)
        
        # gradient 디버깅: epsilon 예측 후 결과 확인
        print(f"[Epsilon Debug] epsilon.requires_grad: {epsilon.requires_grad}, epsilon.grad_fn: {epsilon.grad_fn}")
        if epsilon.grad_fn is None and epsilon.requires_grad:
            print(f"[Epsilon Warning] epsilon has requires_grad=True but grad_fn=None. This indicates a detached tensor!")
        
        # x0 계산
        x0 = (zt - (1 - alpha_prod_t) ** (0.5) * epsilon) / (alpha_prod_t ** (0.5))
        
        # gradient 디버깅: x0 계산 후 결과 확인
        print(f"[X0 Debug] x0.requires_grad: {x0.requires_grad}, x0.grad_fn: {x0.grad_fn}")
        
        return x0
    
    def reset_cache(self):
        """DeepCache 캐시 초기화"""
        if self.deepcache_unet is not None:
            self.deepcache_unet.reset_cache()
        self.prv_features = None 