import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from datasets import load_from_disk, load_dataset
from functools import partial
import logger
from diffusers import AutoencoderKL

from .image_label_guidance import ImageLabelGuidance
from .style_transfer_guidance import StyleTransferGuidance
from .super_resolution import SuperResolution
from .gaussian_deblur import GaussianDeblur
from .molecule_properties import MoleculePropertyGuidance
from .audio_declipping import AduioDeclippingGuidance
from .audio_inpainting import AduioInpaintingGuidance

class BaseGuider:

    def __init__(self, args):
        self.args = args
        self.load_processor()   # vae for latent diffusion
        self.load_guider()      # guidance network

    def load_processor(self):
        # 무조건 diffusers의 AutoencoderKL만 사용
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.args.device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False  # guidance gradient 계산에 필요 없음, 반드시 False로!
        # 기존: self.processor = lambda x: self.vae.decode(x / 0.18215).sample
        self.processor = lambda x: self.vae.decode(x / 0.18215, return_dict=False)[0]

    @torch.enable_grad()
    def process(self, x):
        if hasattr(x, "shape") and x.shape[1] == 4:
            return self.processor(x)
        else:
            return x

    @torch.no_grad()
    def load_guider(self):
        self.get_guidance = None
        device = self.args.device
        guiders = []
        for task, guide_network, target in zip(self.args.tasks, self.args.guide_networks, self.args.targets):
            if task == 'style_transfer':
                guider = StyleTransferGuidance(guide_network, target, device)
            elif task == 'label_guidance':
                guider = ImageLabelGuidance(guide_network, target, device, time=False)
            elif task == 'label_guidance_time':
                guider = ImageLabelGuidance(guide_network, target, device, time=True)
            elif task == 'super_resolution':
                guider = SuperResolution(self.args)
            elif task == 'gaussian_deblur':
                guider = GaussianDeblur(self.args)
            elif task == 'molecule_property':
                guider = MoleculePropertyGuidance(self.args)
            elif task == 'audio_declipping':
                guider = AduioDeclippingGuidance(self.args)
            elif task == 'audio_inpainting':
                guider = AduioInpaintingGuidance(self.args)
            else:
                raise NotImplementedError
            guiders.append(guider)
        if len(guiders) == 1:
            self.get_guidance = partial(guider.get_guidance, post_process=self.process)
        else:
            self.get_guidance = partial(self._get_combined_guidance, guiders=guiders)
        if self.get_guidance is None:
            raise ValueError(f"Unknown guider: {self.args.guider}")
    
    def _get_combined_guidance(self, x, guiders, *args, **kwargs):
        values = []
        for guider in guiders:
            values.append(guider.get_guidance(x, post_process=self.process, *args, **kwargs))
        return sum(values)