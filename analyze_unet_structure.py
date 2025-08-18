#!/usr/bin/env python3
import torch
import torch.nn as nn
import os
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diffusion.unet.openai import create_model
from utils.env_utils import MODEL_PATH

def analyze_unet_structure():
    """CIFAR10 UNet 구조 분석"""
    print("=== CIFAR10 UNet 구조 분석 ===")
    
    # CIFAR10 모델 설정
    image_size = 32
    num_channels = 128
    num_res_blocks = 3
    num_heads = 4
    num_heads_upsample = -1
    num_head_channels = -1
    attention_resolutions = "16,8"
    resblock_updown = False
    learn_sigma = False
    channel_mult = (1, 2, 4, 8)
    
    print(f"모델 설정:")
    print(f"- image_size: {image_size}")
    print(f"- num_channels: {num_channels}")
    print(f"- num_res_blocks: {num_res_blocks}")
    print(f"- channel_mult: {channel_mult}")
    print(f"- attention_resolutions: {attention_resolutions}")
    
    # 모델 생성
    unet = create_model(
        image_size=image_size,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        num_head_channels=num_head_channels,
        attention_resolutions=attention_resolutions,
        resblock_updown=resblock_updown,
        learn_sigma=learn_sigma
    )
    
    print(f"\n=== UNet 구조 분석 ===")
    
    # Input blocks 분석
    print(f"\n1. Input Blocks (인코더):")
    print(f"   총 {len(unet.input_blocks)}개 블록")
    
    for i, block in enumerate(unet.input_blocks):
        if hasattr(block, 'modules'):
            modules = list(block.modules())
            block_type = type(modules[1]).__name__ if len(modules) > 1 else "Unknown"
            print(f"   Block {i}: {block_type}")
            
            # 첫 번째 블록의 출력 채널 수 확인
            if i == 0:
                # 첫 번째 블록은 conv layer
                if hasattr(modules[1], 'out_channels'):
                    print(f"     출력 채널: {modules[1].out_channels}")
            else:
                # ResBlock의 출력 채널 수 확인
                for module in modules:
                    if hasattr(module, 'out_channels'):
                        print(f"     출력 채널: {module.out_channels}")
                        break
    
    # Middle block 분석
    print(f"\n2. Middle Block:")
    middle_modules = list(unet.middle_block.modules())
    print(f"   구성: {len(middle_modules)-1}개 모듈")
    for i, module in enumerate(middle_modules[1:], 1):  # 첫 번째는 Sequential 자체
        print(f"   Module {i}: {type(module).__name__}")
        if hasattr(module, 'out_channels'):
            print(f"     출력 채널: {module.out_channels}")
    
    # Output blocks 분석
    print(f"\n3. Output Blocks (디코더):")
    print(f"   총 {len(unet.output_blocks)}개 블록")
    
    for i, block in enumerate(unet.output_blocks):
        if hasattr(block, 'modules'):
            modules = list(block.modules())
            block_type = type(modules[1]).__name__ if len(modules) > 1 else "Unknown"
            print(f"   Block {i} (U{len(unet.output_blocks)-i}): {block_type}")
            
            # ResBlock의 출력 채널 수 확인
            for module in modules:
                if hasattr(module, 'out_channels'):
                    print(f"     출력 채널: {module.out_channels}")
                    break
    
    # 전체 구조 요약
    print(f"\n=== 구조 요약 ===")
    print(f"Input Blocks: {len(unet.input_blocks)}개")
    print(f"Middle Block: 1개")
    print(f"Output Blocks: {len(unet.output_blocks)}개")
    print(f"총 블록 수: {len(unet.input_blocks) + 1 + len(unet.output_blocks)}개")
    
    # DeepCache 적용 가능한 블록들
    print(f"\n=== DeepCache 적용 가능한 블록들 ===")
    print(f"캐시 가능한 Output Block ID: 1 ~ {len(unet.output_blocks)}")
    print(f"캐시 가능한 Input Block ID: 1 ~ {len(unet.input_blocks)}")
    
    # 실제 모델 로드 테스트
    print(f"\n=== 실제 모델 로드 테스트 ===")
    model_path = os.path.join(MODEL_PATH, 'openai_cifar10.pt')
    if os.path.exists(model_path):
        print(f"모델 파일 존재: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            print(f"모델 로드 성공: {len(state_dict)}개 파라미터")
            
            # 모델에 가중치 로드
            unet.load_state_dict(state_dict)
            print("가중치 로드 성공")
            
            # 테스트 입력으로 forward 테스트
            test_input = torch.randn(1, 3, 32, 32)
            test_timesteps = torch.tensor([999])
            
            with torch.no_grad():
                output = unet(test_input, test_timesteps)
                print(f"Forward 테스트 성공: 입력 {test_input.shape} -> 출력 {output.shape}")
                
        except Exception as e:
            print(f"모델 로드 실패: {e}")
    else:
        print(f"모델 파일 없음: {model_path}")
    
    return unet

if __name__ == "__main__":
    analyze_unet_structure() 