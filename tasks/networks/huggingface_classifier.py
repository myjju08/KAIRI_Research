import torch
from transformers import AutoModelForImageClassification, AutoProcessor
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import Normalize, Compose, Resize

def convert_age_logp(logp: torch.Tensor):
    newlogp = torch.concat([logp[:, :3].mean(dim=1, keepdim=True), logp[:, 5:].mean(dim=1, keepdim=True)], dim=1)
    return newlogp

class HuggingfaceClassifier(nn.Module):

    def __init__(self, targets, guide_network=None):
        
        super(HuggingfaceClassifier, self).__init__()
        
        # torchvision 모델인지 확인
        if guide_network.startswith('torchvision/'):
            import torchvision.models as models
            model_name = guide_network.split('/')[-1]
            
            if model_name == 'resnet18':
                self.model = models.resnet18(pretrained=True)
            elif model_name == 'resnet50':
                self.model = models.resnet50(pretrained=True)
            elif model_name == 'resnet101':
                self.model = models.resnet101(pretrained=True)
            else:
                raise ValueError(f"Unsupported torchvision model: {model_name}")
            
            # torchvision 모델은 processor가 없으므로 기본값 사용
            self.transforms = None
        else:
            # HuggingFace 모델
            self.model = AutoModelForImageClassification.from_pretrained(guide_network)
            processor = AutoProcessor.from_pretrained(guide_network)
            
            # 다양한 processor 형태 처리
            if hasattr(processor, 'size') and isinstance(processor.size, dict):
                # 딕셔너리 형태: {'height': 224, 'width': 224}
                height = processor.size['height']
                width = processor.size['width']
            elif hasattr(processor, 'size') and isinstance(processor.size, (list, tuple)):
                # 리스트/튜플 형태: [224, 224]
                height, width = processor.size
            elif hasattr(processor, 'size') and isinstance(processor.size, int):
                # 정수 형태: 224
                height = width = processor.size
            else:
                # 기본값 (대부분의 ImageNet 모델)
                height = width = 224
            
            # image_mean과 image_std도 안전하게 처리
            if hasattr(processor, 'image_mean') and hasattr(processor, 'image_std'):
                image_mean = processor.image_mean
                image_std = processor.image_std
            else:
                # ImageNet 기본값
                image_mean = [0.485, 0.456, 0.406]
                image_std = [0.229, 0.224, 0.225]
            
            self.transforms = Compose([
                Resize([height, width]),
                Normalize(mean=image_mean, std=image_std)
            ])
        
        self.target = targets

        self.model.eval()

    @torch.enable_grad()
    def forward(self, x):
        '''
            return tensor in the shape of (batch_size, )
        '''
        
        target = self.target

        # torchvision 모델인지 확인
        if self.transforms is None:
            # torchvision 모델: 직접 normalize만 수행
            # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            normalized_x = (x - mean) / std
            
            # torchvision 모델은 .logits 속성이 없음
            logits = self.model(normalized_x)
        else:
            # HuggingFace 모델
            resized_x = self.transforms(x)
            logits = self.model(resized_x).logits

        probs = torch.nn.functional.softmax(logits, dim=1)
        
        log_probs = torch.log(probs)
        
        if isinstance(target, int) or isinstance(target, str):
            selected = log_probs[range(x.size(0)), int(target)]
        else:
            selected = torch.cat([log_probs[range(x.size(0)), _] for _ in target], dim=0)

        return selected


    

        