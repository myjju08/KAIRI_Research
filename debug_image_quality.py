#!/usr/bin/env python3
"""
Debug script to test DiT image generation quality
"""

import torch
import numpy as np
from PIL import Image
import os
import sys

# Add the current directory to the path
sys.path.append('.')

from utils.configs import Arguments
from utils.utils import get_network, get_guidance
from diffusion.ddim import ImageSampler

def test_image_generation():
    """Test image generation with current settings"""
    
    # Create minimal args for testing
    class TestArgs:
        def __init__(self):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.seed = 42
            self.per_sample_batch_size = 4
            self.inference_steps = 50  # Reduced for testing
            self.eta = 0.0
            self.log_traj = False
            self.image_size = 256
            self.num_classes = 1000
            self.model = 'DiT-XL/2'
            self.vae = 'mse'
            self.cfg_scale = 4.0
            self.num_sampling_steps = 50
            self.generators_path = 'DiT/DiT-XL-2-256x256.pt'
            self.data_type = 'image'
            self.logging_dir = './debug_output'
            self.max_show_images = 16
            self.logging_resolution = 256
            self.wandb = False
            self.check_done = False
            self.eval_batch_size = 4
            self.bon_rate = 1
            self.num_samples = 4
            self.log_suffix = ''
    
    args = TestArgs()
    
    # Create output directory
    os.makedirs(args.logging_dir, exist_ok=True)
    
    print("Testing DiT image generation...")
    print(f"Device: {args.device}")
    print(f"Model: {args.model}")
    print(f"VAE: {args.vae}")
    print(f"Image size: {args.image_size}")
    print(f"Inference steps: {args.inference_steps}")
    
    try:
        # Initialize network and guidance
        network = get_network(args)
        guider = get_guidance(args, network)
        
        print("Network and guidance initialized successfully")
        
        # Create sampler
        sampler = ImageSampler(args)
        print("Sampler created successfully")
        
        # Generate samples
        print("Generating samples...")
        samples = sampler.sample(sample_size=args.num_samples, guidance=guider)
        
        print(f"Generated {len(samples)} samples")
        
        # Convert to PIL images
        pil_images = sampler.tensor_to_obj(samples)
        print(f"Converted to {len(pil_images)} PIL images")
        
        # Save individual images for inspection
        for i, img in enumerate(pil_images):
            img_path = os.path.join(args.logging_dir, f"sample_{i:03d}.png")
            img.save(img_path)
            print(f"Saved {img_path}")
            
            # Print image info
            print(f"Image {i}: size={img.size}, mode={img.mode}")
            
            # Convert to array and check statistics
            img_array = np.array(img)
            print(f"Image {i}: min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.2f}, std={img_array.std():.2f}")
        
        # Save grid image
        from torchvision.utils import save_image
        if isinstance(samples, torch.Tensor):
            grid_path = os.path.join(args.logging_dir, "grid.png")
            save_image(samples, grid_path, nrow=2, normalize=True, value_range=(-1, 1))
            print(f"Saved grid image to {grid_path}")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_generation() 