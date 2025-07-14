import torch
import torch.nn as nn
from diffusers import AutoencoderKL
import os

SCALING = 0.18215  # 강제 상수

class VAEProcessor:
    """
    VAE processor for converting between pixel space (256x256) and latent space (32x32)
    Uses the same VAE as DiT: stabilityai/sd-vae-ft-{args.vae}
    """
    def __init__(self, device='cuda', vae_type='mse'):
        self.device = device
        self.vae = None
        self.scale_factor = 8  # 256 / 32 = 8
        self.vae_type = vae_type  # 'ema' or 'mse'
        
    def load_vae(self, vae_path=None):
        """Load VAE model - same as DiT"""
        if vae_path and os.path.exists(vae_path):
            # Load custom VAE
            self.vae = torch.load(vae_path, map_location=self.device)
            print(f'[VAE DEBUG] Loaded custom VAE from {vae_path}')
        else:
            # Load VAE exactly like DiT does
            print(f'[VAE DEBUG] Loading DiT VAE: stabilityai/sd-vae-ft-{self.vae_type}')
            self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{self.vae_type}").to(self.device)
        
        self.vae.eval()
        print('[VAE DEBUG] VAE loaded successfully')
        
    def encode(self, images):
        """
        Encode images to latent space (DiT style)
        Args:
            images: (B, 3, H, W) in [-1, 1] range
        Returns:
            latents: (B, 4, H//8, W//8) in latent space
        """
        if self.vae is None:
            self.load_vae()
        
        # DiT style: encode and scale by 0.18215
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215  # DiT scaling factor
        return latents
    
    def decode(self, latents, preserve_grad=False):
        """
        Decode latents to pixel space (DiT style)
        Args:
            latents: (B, 4, H//8, W//8) in latent space
            preserve_grad: whether to preserve gradients
        Returns:
            images: (B, 3, H, W) in [-1, 1] range
        """
        if self.vae is None:
            self.load_vae()
        
        # Ensure latents are float32 to match VAE model
        latents = latents.to(dtype=torch.float32)
        
        # DiT style: scale by 1/0.18215 before decoding
        latents_scaled = latents / 0.18215  # DiT scaling factor
        
        if preserve_grad:
            # For gradient computation
            images = self.vae.decode(latents_scaled).sample
        else:
            # For final output
            with torch.no_grad():
                images = self.vae.decode(latents_scaled).sample
        
        return images

def get_diffusion_with_vae(args):
    """
    Modified get_diffusion function that includes VAE processing
    Uses the same VAE as DiT: stabilityai/sd-vae-ft-{args.vae}
    """
    from .openai import get_diffusion
    
    # Get the original diffusion components
    model, ts, alpha_prod_ts, alpha_prod_t_prevs = get_diffusion(args)
    
    # Create VAE processor with args.vae (same as DiT)
    vae_type = getattr(args, 'vae', 'mse')  # Default to 'mse' like DiT
    vae_processor = VAEProcessor(device=args.device, vae_type=vae_type)
    vae_processor.load_vae()
    
    # Create a wrapper model that handles VAE encoding/decoding
    class DiTWithVAE(nn.Module):
        def __init__(self, dit_model, vae_processor):
            super().__init__()
            self.dit_model = dit_model
            self.vae_processor = vae_processor
            
        def forward(self, x, t, y):
            """
            Forward pass through DiT model only (no VAE encoding/decoding)
            This matches DiT's sampling process where VAE is handled separately
            """
            # x is already in latent space (4 channels)
            # Just pass through DiT model
            return self.dit_model(x, t, y)
    
    # Wrap the model
    wrapped_model = DiTWithVAE(model, vae_processor)
    
    return wrapped_model, ts, alpha_prod_ts, alpha_prod_t_prevs 