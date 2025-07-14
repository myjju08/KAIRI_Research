import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from diffusers.utils import randn_tensor
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class VAEHandler:
    """
    VAE를 사용하여 256x256 이미지와 32x32 latent space 간의 변환을 처리합니다.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.vae = None
        self.scale_factor = 8  # 256 / 32 = 8
        
    def load_vae(self, vae_path=None):
        """
        VAE 모델을 로드합니다. vae_path가 None이면 기본 VAE를 사용합니다.
        """
        if vae_path:
            # 커스텀 VAE 로드
            self.vae = torch.load(vae_path, map_location=self.device)
        else:
            # 기본 VAE 로드 (Stable Diffusion VAE)
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float16
            ).to(self.device)
        
        self.vae.eval()
        
    def encode(self, images):
        """
        256x256 이미지를 32x32 latent로 인코딩합니다.
        
        Args:
            images: (B, 3, 256, 256) tensor in [-1, 1] range
            
        Returns:
            latents: (B, 4, 32, 32) tensor
        """
        if self.vae is None:
            raise ValueError("VAE가 로드되지 않았습니다. load_vae()를 먼저 호출하세요.")
            
        with torch.no_grad():
            # VAE 인코딩
            latents = self.vae.encode(images).latent_dist.sample()
            # 스케일링 (VAE의 스케일 팩터 적용)
            latents = latents * self.vae.config.scaling_factor
            
        return latents
    
    def decode(self, latents):
        """
        32x32 latent를 256x256 이미지로 디코딩합니다.
        
        Args:
            latents: (B, 4, 32, 32) tensor
            
        Returns:
            images: (B, 3, 256, 256) tensor in [-1, 1] range
        """
        if self.vae is None:
            raise ValueError("VAE가 로드되지 않았습니다. load_vae()를 먼저 호출하세요.")
            
        with torch.no_grad():
            # 스케일링 되돌리기
            latents = latents / self.vae.config.scaling_factor
            # VAE 디코딩
            images = self.vae.decode(latents).sample
            
        return images
    
    def encode_noise(self, noise):
        """
        노이즈를 latent space로 변환합니다.
        
        Args:
            noise: (B, 3, 256, 256) tensor
            
        Returns:
            latent_noise: (B, 4, 32, 32) tensor
        """
        return self.encode(noise)
    
    def decode_samples(self, samples):
        """
        샘플링된 latent를 이미지로 변환합니다.
        
        Args:
            samples: (B, 4, 32, 32) tensor
            
        Returns:
            images: (B, 3, 256, 256) tensor
        """
        return self.decode(samples)

def create_latent_noise(batch_size, device, vae_handler):
    """
    Latent space에서 노이즈를 생성합니다.
    
    Args:
        batch_size: 배치 크기
        device: 디바이스
        vae_handler: VAE 핸들러
        
    Returns:
        latent_noise: (B, 4, 32, 32) tensor
    """
    # 256x256 노이즈 생성
    noise = randn_tensor(
        shape=(batch_size, 3, 256, 256),
        device=device
    )
    
    # Latent space로 변환
    latent_noise = vae_handler.encode_noise(noise)
    
    return latent_noise

def convert_latent_to_image(latents, vae_handler):
    """
    Latent를 이미지로 변환합니다.
    
    Args:
        latents: (B, 4, 32, 32) tensor
        vae_handler: VAE 핸들러
        
    Returns:
        images: PIL 이미지 리스트
    """
    # 디코딩
    images = vae_handler.decode_samples(latents)
    
    # [-1, 1] -> [0, 1] 변환
    images = (images / 2 + 0.5).clamp(0, 1)
    
    # 텐서를 PIL 이미지로 변환
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    
    pil_images = [Image.fromarray(image) for image in images]
    
    return pil_images 