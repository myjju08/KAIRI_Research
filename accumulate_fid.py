#!/usr/bin/env python3
"""
여러 실험의 데이터를 합쳐서 FID를 측정하는 스크립트
"""

import os
import glob
import numpy as np
from PIL import Image
import argparse
from evaluations.image import ImageEvaluator
from utils.utils import get_config
from logger import load_samples

def accumulate_samples_from_experiments(base_dir, target_class=6):
    """
    여러 실험 디렉토리에서 샘플들을 수집합니다.
    
    Args:
        base_dir: 실험들이 저장된 기본 디렉토리
        target_class: 타겟 클래스 (CIFAR-10에서 frog=6)
    
    Returns:
        all_samples: 수집된 모든 샘플들의 리스트
    """
    all_samples = []
    experiment_dirs = glob.glob(os.path.join(base_dir, "*"))
    
    print(f"Searching for experiments in: {base_dir}")
    print(f"Found directories: {experiment_dirs}")
    
    for exp_dir in experiment_dirs:
        if os.path.isdir(exp_dir):
            try:
                # images.npy 파일 찾기
                npy_path = os.path.join(exp_dir, "images.npy")
                if os.path.exists(npy_path):
                    # numpy 배열 로드
                    images_array = np.load(npy_path)
                    print(f"Loaded {len(images_array)} samples from {exp_dir}")
                    
                    # PIL Image로 변환
                    for img_array in images_array:
                        img = Image.fromarray(img_array)
                        all_samples.append(img)
                        
            except Exception as e:
                print(f"Error loading from {exp_dir}: {e}")
    
    print(f"Total accumulated samples: {len(all_samples)}")
    return all_samples

def evaluate_accumulated_samples(samples, args):
    """
    수집된 샘플들을 평가합니다.
    
    Args:
        samples: 평가할 샘플들
        args: 설정 인자들
    
    Returns:
        metrics: 평가 결과
    """
    if len(samples) == 0:
        print("No samples to evaluate!")
        return None
    
    # ImageEvaluator 생성
    evaluator = ImageEvaluator(args)
    
    # 평가 실행
    print(f"Evaluating {len(samples)} accumulated samples...")
    metrics = evaluator.evaluate(samples)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Accumulate samples from multiple experiments and compute FID')
    parser.add_argument('--base_dir', type=str, default='logs', 
                       help='Base directory containing experiment folders')
    parser.add_argument('--target_class', type=int, default=6,
                       help='Target class (6 for frog in CIFAR-10)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # 기본 설정 생성
    config_args = get_config()
    config_args.device = args.device
    config_args.dataset = 'cifar10'
    config_args.targets = [args.target_class]
    config_args.eval_batch_size = 512
    
    print(f"=== ACCUMULATED FID EVALUATION ===")
    print(f"Base directory: {args.base_dir}")
    print(f"Target class: {args.target_class}")
    print(f"Device: {args.device}")
    
    # 샘플 수집
    all_samples = accumulate_samples_from_experiments(args.base_dir, args.target_class)
    
    if len(all_samples) == 0:
        print("No samples found! Please check the base directory.")
        return
    
    # 평가 실행
    metrics = evaluate_accumulated_samples(all_samples, config_args)
    
    if metrics is not None:
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Total samples evaluated: {len(all_samples)}")
        
        if 'fid' in metrics:
            print(f"FID Score: {metrics['fid']:.4f}")
        
        if 'inception_score' in metrics:
            print(f"Inception Score: {metrics['inception_score']:.4f}")
        
        if 'validity' in metrics:
            print(f"Validity: {metrics['validity']:.4f}")
        
        print(f"All metrics: {metrics}")
    else:
        print("Evaluation failed!")

if __name__ == '__main__':
    main() 