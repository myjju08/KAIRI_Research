#!/usr/bin/env python3
"""
기존 Training-Free-Guidance의 FID 계산 시스템을 사용해서 정확한 FID 계산
"""

import os
import sys
import numpy as np
from PIL import Image
import torch

# 현재 디렉토리를 Python 경로에 추가
sys.path.append('.')

def load_combined_images(images_path="logs/combined/images.npy"):
    """합쳐진 이미지들을 로드합니다."""
    if not os.path.exists(images_path):
        print(f"이미지 파일을 찾을 수 없습니다: {images_path}")
        return None
    
    try:
        images_array = np.load(images_path)
        images = [Image.fromarray(img) for img in images_array]
        print(f"로드된 이미지 수: {len(images)}")
        return images
    except Exception as e:
        print(f"이미지 로드 중 오류 발생: {e}")
        return None

def load_images_from_guidance_scale(guidance_scale, num_runs=5, base_logging_dir="logs"):
    """특정 guidance_scale에서 생성된 이미지들을 로드합니다."""
    all_images = []
    
    for i in range(num_runs):
        # guidance_scale이 정수인 경우와 실수인 경우를 모두 처리
        if guidance_scale == int(guidance_scale):
            scale_str = str(int(guidance_scale))
        else:
            scale_str = str(guidance_scale)
        
        # start_gradient 값 추출 (디렉토리 구조에서)
        start_gradient = None
        if base_logging_dir != "logs":
            # logs/start_gradient_X/ 형태에서 X 추출
            dir_parts = base_logging_dir.split('/')
            for part in dir_parts:
                if part.startswith('start_gradient_'):
                    start_gradient = part.replace('start_gradient_', '')
                    break
        
        if start_gradient:
            run_dir = os.path.join(base_logging_dir, f"guidance_scale_{scale_str}/run_{i}")
        else:
            # 기존 구조 호환성 유지
            run_dir = os.path.join(base_logging_dir, f"guidance_scale_{scale_str}/run_{i}")
        
        print(f"Guidance Scale {guidance_scale}, 실행 {i}에서 이미지 로드 중: {run_dir}")
        
        # find 명령어로 images.npy 파일 찾기
        import subprocess
        try:
            result = subprocess.run(['find', run_dir, '-name', 'images.npy', '-type', 'f'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0 or not result.stdout.strip():
                print(f"경고: {run_dir}에서 images.npy 파일을 찾을 수 없습니다.")
                continue
            
            # 첫 번째 이미지 파일 경로 사용
            images_npy_path = result.stdout.strip().split('\n')[0]
            print(f"발견된 이미지 파일: {images_npy_path}")
            
            # 파일이 실제로 존재하는지 확인
            if not os.path.exists(images_npy_path):
                print(f"경고: 파일이 존재하지 않습니다: {images_npy_path}")
                continue
            
            images_array = np.load(images_npy_path)
            images = [Image.fromarray(img) for img in images_array]
            all_images.extend(images)
            
            print(f"Guidance Scale {guidance_scale}, 실행 {i}에서 {len(images)}개 이미지 로드됨")
        except Exception as e:
            print(f"이미지 로드 중 오류 발생: {e}")
            continue
    
    print(f"Guidance Scale {guidance_scale}: 총 {len(all_images)}개 이미지가 로드되었습니다.")
    return all_images

def save_combined_images(images, output_dir="logs/combined"):
    """합쳐진 이미지들을 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)
    
    # numpy 배열로 변환하여 저장
    images_array = np.stack([np.array(img) for img in images])
    np.save(os.path.join(output_dir, "images.npy"), images_array)
    
    print(f"합쳐진 이미지들이 {output_dir}/images.npy에 저장되었습니다.")
    return output_dir

def calculate_fid_proper(images, dataset="cifar10"):
    """기존 시스템의 FID 계산 방법을 사용"""
    print("기존 시스템으로 FID 계산 중...")
    
    try:
        from evaluations.utils.fid import calculate_fid
        from tasks.utils import load_image_dataset
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 중인 디바이스: {device}")
        
        # 참조 이미지들 로드 (CIFAR-10 테스트셋)
        print("참조 이미지들을 로드 중...")
        ref_images = load_image_dataset(dataset, num_samples=-1, target=-1, return_tensor=False)
        print(f"참조 이미지 수: {len(ref_images)}")
        
        # FID 계산
        batch_size = 512
        fid_score = calculate_fid(
            ref=ref_images,
            test=images,
            batch_size=batch_size,
            device=device,
            dims=2048,
            num_workers=1,
            cache_path=None
        )
        
        print(f"FID Score: {fid_score:.4f}")
        return {"fid": fid_score}
        
    except Exception as e:
        print(f"FID 계산 중 오류 발생: {e}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="기존 Training-Free-Guidance의 FID 계산 시스템을 사용해서 정확한 FID 계산")
    parser.add_argument("--guidance_scale", type=float, default=None, help="특정 guidance_scale의 이미지들을 합쳐서 FID 계산")
    parser.add_argument("--num_runs", type=int, default=5, help="각 guidance_scale당 실행 횟수")
    parser.add_argument("--base_logging_dir", type=str, default="logs", help="로그 디렉토리 기본 경로")
    parser.add_argument("--dataset", type=str, default="cifar10", help="데이터셋 이름")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("기존 시스템으로 정확한 FID 계산")
    print("=" * 50)
    
    if args.guidance_scale is not None:
        # 특정 guidance_scale의 이미지들 로드
        images = load_images_from_guidance_scale(args.guidance_scale, args.num_runs, args.base_logging_dir)
        
        # start_gradient 값 추출
        start_gradient = None
        if args.base_logging_dir != "logs":
            dir_parts = args.base_logging_dir.split('/')
            for part in dir_parts:
                if part.startswith('start_gradient_'):
                    start_gradient = part.replace('start_gradient_', '')
                    break
        
        if start_gradient:
            output_dir = f"logs/start_gradient_{start_gradient}/guidance_scale_{args.guidance_scale}_combined"
        else:
            output_dir = f"logs/guidance_scale_{args.guidance_scale}_combined"
    else:
        # 기존 방식: combined 이미지 로드
        images = load_combined_images()
        output_dir = "logs/combined"
    
    if images is None:
        print("이미지를 로드할 수 없습니다.")
        return
    
    # 합쳐진 이미지 저장
    output_dir = save_combined_images(images, output_dir)
    
    # FID score 계산
    metrics = calculate_fid_proper(images, args.dataset)
    
    if metrics:
        print("\n" + "=" * 50)
        print("FID 평가 결과")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # 결과를 JSON 파일로 저장
        import json
        if args.guidance_scale is not None:
            # start_gradient 값 추출
            start_gradient = None
            if args.base_logging_dir != "logs":
                dir_parts = args.base_logging_dir.split('/')
                for part in dir_parts:
                    if part.startswith('start_gradient_'):
                        start_gradient = part.replace('start_gradient_', '')
                        break
            
            if start_gradient:
                result_file = f"logs/start_gradient_{start_gradient}/guidance_scale_{args.guidance_scale}_combined/fid_results_proper.json"
            else:
                result_file = f"logs/guidance_scale_{args.guidance_scale}_combined/fid_results_proper.json"
        else:
            result_file = "logs/combined/fid_results_proper.json"
        
        with open(result_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n결과가 {result_file}에 저장되었습니다.")
    else:
        print("FID score 계산에 실패했습니다.")
    
    print("\n" + "=" * 50)
    print("완료!")
    print("=" * 50)

if __name__ == "__main__":
    main() 