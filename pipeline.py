import torch
import os
import numpy as np
import PIL.Image as Image
from abc import ABC, abstractmethod
from diffusion.base import BaseSampler
from methods.base import BaseGuidance
from evaluations.base import BaseEvaluator
from utils.configs import Arguments
import logger

class BasePipeline(object):
    def __init__(self,
                 args: Arguments, 
                 network: BaseSampler, 
                 guider: BaseGuidance, 
                 evaluator: BaseEvaluator,
                 bon_guider=None):
        self.network = network
        self.guider = guider
        self.evaluator = evaluator
        self.logging_dir = args.logging_dir
        self.check_done = args.check_done
        
        self.bon_rate = args.bon_rate
        self.batch_size = args.eval_batch_size
        
        self.bon_guider = bon_guider if bon_guider is not None else self.guider
        
        # sampler에 포함된 VAE를 guidance 객체에 공유하여 DiT/DPS에서 decode 사용 가능하도록 설정
        if hasattr(network, 'vae') and getattr(network, 'vae', None) is not None:
            if hasattr(self.guider, 'vae'):
                self.guider.vae = network.vae
            else:
                setattr(self.guider, 'vae', network.vae)
            if self.bon_guider is not None and self.bon_guider is not self.guider:
                if hasattr(self.bon_guider, 'vae'):
                    self.bon_guider.vae = network.vae
                else:
                    setattr(self.bon_guider, 'vae', network.vae)
        
    @abstractmethod
    def sample(self, sample_size: int):
        
        samples = self.check_done_and_load_sample()
        
        if samples is None:

            guidance_batch_size = self.batch_size  

            # For DiT models, we need to handle the case where bon_rate=1
            if self.bon_rate == 1:
                # Direct sampling without multiple candidates
                samples = self.network.sample(sample_size=sample_size, guidance=self.guider)
            else:
                # Original logic for multiple candidates
                samples = self.network.sample(sample_size=sample_size * self.bon_rate, guidance=self.guider)

                logp_list = []
                for i in range(0, samples.shape[0], guidance_batch_size):
                    batch_samples = samples[i:i + guidance_batch_size]
                    batch_logp = self.bon_guider.guider.get_guidance(batch_samples, return_logp=True, check_grad=False)
                    logp_list.append(batch_logp)

                logp = torch.cat(logp_list, dim=0).view(-1)

                samples = samples.view(sample_size, int(self.bon_rate), *samples.shape[1:])
                logp = logp.view(sample_size, int(self.bon_rate))

                idx = logp.argmax(dim=1)
                arange_idx = torch.arange(sample_size, device=samples.device)
                idx = idx.to(samples.device)
                samples = samples[arange_idx, idx]

            samples = self.network.tensor_to_obj(samples)
                    
        return samples
    
    def evaluate(self, samples):
        return self.check_done_and_evaluate(samples)
    
    def check_done_and_evaluate(self, samples):
        if self.check_done and os.path.exists(os.path.join(self.logging_dir, 'metrics.json')):
            logger.log("Metrics already generated. To regenerate, please set `check_done` to `False`.")
            return None
        return self.evaluator.evaluate(samples)

    def check_done_and_load_sample(self):
        if self.check_done and os.path.exists(os.path.join(self.logging_dir, "finished_sampling")):
            logger.log("found tags for generated samples, should load directly. To regenerate, please set `check_done` to `False`.")
            return logger.load_samples()

        return None