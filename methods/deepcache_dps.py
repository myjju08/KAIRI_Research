import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .dps import DPSGuidance
from models.deepcache_unet import create_deepcache_unet
import time
import os
import json


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
		self.cache_block_id = getattr(args, 'cache_block_id', None)
		
		# model_type 설정 (부모 클래스에서 상속)
		self.model_type = getattr(args, 'model_type', 'unet')
		
		# DeepCache UNet 초기화 (나중에 모델 로드 시 적용)
		self.deepcache_unet = None
		self.original_model = None  # 원본 모델 저장용
		
		# 캐시 관련 변수들
		self.prv_features = None
		# gradient / cross-entropy 기록 구조
		self.grad_updates_per_step = []  # Deprecated: 유지만 하고 사용하지 않음 (호환용)
		self.grad_norms_per_step = []  # list of dict: { 'step': int, 'grad_norms': List[float] }
		self.cross_entropy_per_step = []  # list of dict: { 'step': int, 'cross_entropy': List[float] }
		self.sample_histories = {}  # dict: sample_idx -> List[Dict[str, Any]]
		self._gradient_log_initialized = False
		self._sample_metrics_initialized = False
		log_dir = os.path.abspath(os.path.expanduser(getattr(args, 'logging_dir', '.')))
		os.makedirs(log_dir, exist_ok=True)
		target = getattr(args, 'target', 'unknown')
		guidance_name = getattr(args, 'guidance_name', 'dps')
		timestamp = time.strftime('%Y%m%d_%H%M%S')
		self.gradient_log_path = os.path.join(
			log_dir,
			f"gradient_norms_guidance={guidance_name}_target={target}_{timestamp}.jsonl"
		)
		self.sample_metrics_log_path = os.path.join(
			log_dir,
			f"sample_metrics_guidance={guidance_name}_target={target}_{timestamp}.jsonl"
		)
		
		print(f"[DeepCache] DeepCacheDPSGuidance 초기화: use_deepcache={self.use_deepcache}, cache_interval={self.cache_interval}")
	
	def _setup_deepcache_model(self, model: torch.nn.Module) -> torch.nn.Module:
		"""DeepCache 모델 설정"""
		if self.deepcache_unet is None:
			self.deepcache_unet = create_deepcache_unet(
				original_unet=model,
				cache_interval=self.cache_interval,
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
		
		print(f"[DeepCache DPS Debug] _guide_step_unet 시작: t={t}, current_t={current_t}, cache_interval={self.cache_interval}, clean_step={getattr(self.args, 'clean_step', None)}")
		# 로깅용 step index 계산 (원본 timestep 유지)
		if isinstance(t, int):
			step_index = int(t)
		elif isinstance(t, torch.Tensor):
			if t.dim() == 0:
				step_index = int(t.item())
			else:
				step_index = int(t[0].item())
		else:
			step_index = int(t)
		
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
			print(f"[DeepCache DPS] 캐시 업데이트 스텝 (t={current_t}): x0 예측에서 전체 계산")
		else:
			print(f"[DeepCache DPS] 캐싱 스텝 (t={current_t}): x0 예측에서 캐시 사용")
		
		# DeepCache를 사용한 x0 예측 함수 (진단용 토글 지원)
		def func(zt):
			use_original_for_parity = os.environ.get('DC_PARITY_USE_ORIGINAL', '0') == '1'
			if use_original_for_parity:
				# 원본 UNet 경로로 x0 계산 (파리티 진단용)
				return (zt - (1 - alpha_prod_t) ** (0.5) * model(zt, t_tensor)) / (alpha_prod_t ** (0.5))
			# DeepCache 모델로 x0 예측
			x0 = self._deepcache_x0_predict(zt, deepcache_model, t_tensor, alpha_prod_t)
			return x0
		
		# gradient 디버깅: guidance 계산 전 입력값 확인
		x_for_grad = x.clone().detach().requires_grad_()
		print(f"[DeepCache DPS Debug] t={t}, x_for_grad.requires_grad: {x_for_grad.requires_grad}")
		
		guidance = self.guider.get_guidance(
			x_for_grad, 
			func, 
			**kwargs
		)
		
		per_sample_grad_norms = None
		per_sample_cross_entropy = None
		# gradient 디버깅: guidance 결과 확인
		if guidance is not None:
			grad_norm = torch.norm(guidance).item()
			grad_max = torch.max(torch.abs(guidance)).item()
			print(f"[DeepCache DPS Debug] t={t}, grad_norm: {grad_norm:.6f}, grad_max: {grad_max:.6f}")
			try:
				if guidance.dim() == 0:
					per_sample_grad_norms = [float(torch.abs(guidance).item())]
				else:
					per_sample_grad_norms = guidance.view(guidance.shape[0], -1).norm(dim=1).detach().float().cpu().tolist()
			except Exception as e:
				per_sample_grad_norms = None
				print(f"[DeepCache DPS Warning] per-sample gradient norm logging failed at t={t}: {e}")
			# gradient가 거의 0인지 확인
			if grad_norm < 1e-8:
				print(f"[DeepCache DPS Warning] t={t}, Gradient is almost zero! This might be a caching step.")
		else:
			print(f"[DeepCache DPS Debug] t={t}, guidance is None")
		
		# follow the schedule of DPS paper
		logp_norm = self.guider.get_guidance(x.clone().detach(), func, return_logp=True, check_grad=False, **kwargs)
		try:
			if isinstance(logp_norm, torch.Tensor):
				ce_tensor = (-logp_norm).detach().float()
				if ce_tensor.ndim > 1:
					ce_tensor = ce_tensor.view(ce_tensor.shape[0], -1).mean(dim=1)
				per_sample_cross_entropy = ce_tensor.cpu().tolist()
				if not isinstance(per_sample_cross_entropy, list):
					per_sample_cross_entropy = [per_sample_cross_entropy]
			elif logp_norm is not None:
				per_sample_cross_entropy = [float(-logp_norm)]
		except Exception as e:
			per_sample_cross_entropy = None
			print(f"[DeepCache DPS Debug] cross entropy logging failed at t={t}: {e}")
		if per_sample_cross_entropy is not None and len(per_sample_cross_entropy) > 0:
			mean_ce = sum(per_sample_cross_entropy) / len(per_sample_cross_entropy)
			print(f"[DeepCache DPS Debug] t={t}, cross_entropy: {mean_ce:.6f}")
		if per_sample_grad_norms is not None or per_sample_cross_entropy is not None:
			self._log_gradient_statistics(
				step_index=step_index,
				grad_norms=per_sample_grad_norms,
				current_t=current_t,
				is_cache_update_step=is_cache_update_step,
				cross_entropies=per_sample_cross_entropy
			)
		
		# guidance strength 스케줄 적용 (기본값은 args.guidance_strength)
		guidance_strength = getattr(self.args, 'guidance_strength', 1.0)
		if (
			hasattr(self.args, 'guidance_scale_schedule') and
			hasattr(self.args, 'guidance_transition_steps') and
			self.args.guidance_scale_schedule is not None and
			self.args.guidance_transition_steps is not None
		):
			if 't_idx' in kwargs:
				current_step = int(kwargs['t_idx'])
			else:
				current_step = int(step_index)
			guidance_strength = self._get_multi_step_guidance_scale(
				current_step,
				guidance_strength,
				self.args.guidance_scale_schedule,
				self.args.guidance_transition_steps,
			)
		elif (
			hasattr(self.args, 'guidance_early') and
			hasattr(self.args, 'guidance_late') and
			hasattr(self.args, 'guidance_transition_step') and
			self.args.guidance_early is not None and
			self.args.guidance_late is not None and
			self.args.guidance_transition_step is not None
		):
			if 't_idx' in kwargs:
				current_step = int(kwargs['t_idx'])
			else:
				if isinstance(t_tensor, torch.Tensor):
					if t_tensor.dim() == 0:
						current_step = int(t_tensor.item())
					else:
						current_step = int(t_tensor.view(-1)[0].item())
				else:
					current_step = int(t_tensor)
			guidance_strength = (
				self.args.guidance_late
				if current_step >= self.args.guidance_transition_step
				else self.args.guidance_early
			)

		x_prev = x_prev + guidance_strength * guidance / torch.abs(logp_norm.view(-1, * ([1] * (len(x_prev.shape) - 1)) ))
			
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
	
	def _log_gradient_statistics(self, step_index, grad_norms=None, current_t=None, is_cache_update_step=None, cross_entropies=None):
		record = {'step': int(step_index)}
		if current_t is not None:
			record['current_t'] = int(current_t)
		if is_cache_update_step is not None:
			record['cache_update_step'] = bool(is_cache_update_step)
		samples_data = []
		max_len = 0
		if isinstance(grad_norms, list):
			max_len = max(max_len, len(grad_norms))
		if isinstance(cross_entropies, list):
			max_len = max(max_len, len(cross_entropies))
		if max_len > 0:
			for sample_idx in range(max_len):
				sample_entry = {'sample': sample_idx}
				grad_value = None
				ce_value = None
				if isinstance(grad_norms, list) and sample_idx < len(grad_norms):
					grad_value = grad_norms[sample_idx]
					sample_entry['grad_norm'] = grad_value
				if isinstance(cross_entropies, list) and sample_idx < len(cross_entropies):
					ce_value = cross_entropies[sample_idx]
					sample_entry['cross_entropy'] = ce_value
				if grad_value is not None or ce_value is not None:
					samples_data.append(sample_entry)
					sample_history_entry = {
						'step': int(step_index),
						'grad_norm': grad_value,
						'cross_entropy': ce_value
					}
					if current_t is not None:
						sample_history_entry['current_t'] = int(current_t)
					self.sample_histories.setdefault(sample_idx, []).append(sample_history_entry)
		if samples_data:
			record['samples'] = samples_data
			if isinstance(grad_norms, list):
				self.grad_norms_per_step.append({
					'step': int(step_index),
					'grad_norms': grad_norms,
				})
			if isinstance(cross_entropies, list):
				self.cross_entropy_per_step.append({
					'step': int(step_index),
					'cross_entropy': cross_entropies,
				})
		self._write_gradient_record(record)
		if samples_data:
			self._write_sample_records(samples_data, step_index=step_index, current_t=current_t)

	def _write_gradient_record(self, record):
		if not hasattr(self, 'gradient_log_path') or self.gradient_log_path is None:
			return
		try:
			dirname = os.path.dirname(self.gradient_log_path)
			if dirname:
				os.makedirs(dirname, exist_ok=True)
			mode = 'a'
			if not self._gradient_log_initialized:
				mode = 'w'
			with open(self.gradient_log_path, mode, encoding='utf-8') as f:
				if not self._gradient_log_initialized:
					meta = {
						'guidance_name': getattr(self.args, 'guidance_name', 'dps'),
						'target': getattr(self.args, 'target', None),
						'use_deepcache': self.use_deepcache,
						'cache_interval': self.cache_interval,
						'cache_block_id': self.cache_block_id,
						'model_type': self.model_type,
						'iter_steps': getattr(self.args, 'iter_steps', None),
						'guidance_strength': getattr(self.args, 'guidance_strength', None),
						'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
					}
					f.write(json.dumps({'meta': meta}) + "\n")
					self._gradient_log_initialized = True
				f.write(json.dumps(record) + "\n")
		except Exception as e:
			print(f"[DeepCache DPS Warning] Failed to log gradient statistics: {e}")

	def _write_sample_records(self, samples, step_index, current_t=None):
		if not hasattr(self, 'sample_metrics_log_path') or self.sample_metrics_log_path is None:
			return
		if not samples:
			return
		try:
			dirname = os.path.dirname(self.sample_metrics_log_path)
			if dirname:
				os.makedirs(dirname, exist_ok=True)
			mode = 'a'
			if not self._sample_metrics_initialized:
				mode = 'w'
			with open(self.sample_metrics_log_path, mode, encoding='utf-8') as f:
				if not self._sample_metrics_initialized:
					meta = {
						'guidance_name': getattr(self.args, 'guidance_name', 'dps'),
						'target': getattr(self.args, 'target', None),
						'use_deepcache': self.use_deepcache,
						'cache_interval': self.cache_interval,
						'cache_block_id': self.cache_block_id,
						'model_type': self.model_type,
						'iter_steps': getattr(self.args, 'iter_steps', None),
						'guidance_strength': getattr(self.args, 'guidance_strength', None),
						'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
					}
					f.write(json.dumps({'meta': meta}) + "\n")
					self._sample_metrics_initialized = True
				for sample in samples:
					entry = {
						'step': int(step_index),
						'sample': int(sample.get('sample', 0)),
						'grad_norm': sample.get('grad_norm'),
						'cross_entropy': sample.get('cross_entropy')
					}
					if current_t is not None:
						entry['current_t'] = int(current_t)
					f.write(json.dumps(entry) + "\n")
		except Exception as e:
			print(f"[DeepCache DPS Warning] Failed to log sample metrics: {e}")
	
	def reset_cache(self):
		"""DeepCache 캐시 초기화"""
		if self.deepcache_unet is not None:
			self.deepcache_unet.reset_cache()
		self.prv_features = None 