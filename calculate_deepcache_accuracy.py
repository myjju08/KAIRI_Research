#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from tasks.networks.resnet import ResNet18
from utils.configs import Arguments
from logger import setup_logger

def load_images_from_deepcache_experiment(exp_dir):
    """Load images from a DeepCache experiment directory"""
    all_images = []
    
    # DeepCache 실험 디렉토리 구조: cache_X_clean_Y/guidance_name=.../model=.../guide_net=.../target=.../bon=.../guidance_strength=...
    # images.npy 파일 찾기
    images_file = None
    for root, dirs, files in os.walk(exp_dir):
        if "images.npy" in files:
            images_file = os.path.join(root, "images.npy")
            break
    
    if not images_file:
        print(f"Images file not found in: {exp_dir}")
        return []
    
    print(f"Loading images from: {images_file}")
    try:
        # Load numpy array
        img_array = np.load(images_file)
        print(f"Loaded array shape: {img_array.shape}")
        
        # img_array는 (num_samples, height, width, channels) 형태
        for i in range(img_array.shape[0]):
            img = Image.fromarray(img_array[i].astype(np.uint8))
            all_images.append(img)
        
        print(f"Loaded {img_array.shape[0]} images from {exp_dir}")
        
    except Exception as e:
        print(f"Error loading {images_file}: {e}")
        return []
    
    return all_images

def calculate_condition_accuracy_for_deepcache_experiment(exp_dir, target_class=2):
    """Calculate condition accuracy for a specific DeepCache experiment"""
    try:
        # Load generated images
        generated_images = load_images_from_deepcache_experiment(exp_dir)
        if not generated_images:
            print(f"No images found for experiment: {exp_dir}")
            return None
        
        # Setup configuration
        config_args = Arguments()
        config_args.targets = [str(target_class)]
        config_args.tasks = ['classifier']
        config_args.eval_batch_size = 32
        config_args.image_size = 32
        config_args.classifiers_path = "tasks/classifiers"
        config_args.args_classifiers_path = "tasks/classifiers"
        config_args.logging_dir = "logs"
        config_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logger
        setup_logger(config_args)
        
        # ResNet18 classifier 로드
        classifier = ResNet18(targets=target_class, guide_network='resnet_cifar10.pt')
        classifier = classifier.to(config_args.device)
        classifier.eval()
        
        # Convert images to tensors
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        image_tensors = []
        for img in generated_images:
            tensor = transform(img)
            image_tensors.append(tensor)
        
        # Stack tensors
        batch = torch.stack(image_tensors).to(config_args.device)
        
        # Calculate accuracy
        with torch.no_grad():
            outputs = classifier(batch)  # target class에 대한 log probability 반환
            probabilities = torch.exp(outputs)  # log probability를 probability로 변환
            accuracy = probabilities.mean().item()
        
        print(f"Condition accuracy: {accuracy:.4f}")
        return accuracy
        
    except Exception as e:
        print(f"Error calculating condition accuracy for {exp_dir}: {e}")
        return None

def process_deepcache_experiments(experiment_base_dir):
    """Process all DeepCache experiments"""
    print(f"=== Processing DeepCache experiments from: {experiment_base_dir} ===")
    
    if not os.path.exists(experiment_base_dir):
        print(f"Experiment base directory not found: {experiment_base_dir}")
        return []
    
    results = []
    
    # 실험 디렉토리 찾기 (deepcache_experiments_YYYYMMDD_HHMMSS 형태)
    experiment_dirs = []
    for item in os.listdir(experiment_base_dir):
        if item.startswith("deepcache_experiments_"):
            experiment_dirs.append(os.path.join(experiment_base_dir, item))
    
    if not experiment_dirs:
        print(f"No DeepCache experiment directories found in: {experiment_base_dir}")
        return []
    
    # 가장 최근 실험 디렉토리 사용 (또는 모든 디렉토리 처리)
    for exp_dir in experiment_dirs:
        print(f"\nProcessing experiment directory: {exp_dir}")
        
        # 각 실험 서브디렉토리 찾기 (cache_X_clean_Y 형태)
        for item in os.listdir(exp_dir):
            if item.startswith("cache_") and os.path.isdir(os.path.join(exp_dir, item)):
                cache_clean_dir = os.path.join(exp_dir, item)
                
                # cache_X_clean_Y에서 X와 Y 추출
                try:
                    parts = item.split("_")
                    cache_interval = int(parts[1])
                    clean_step = int(parts[3])
                    
                    print(f"\nProcessing: cache_interval={cache_interval}, clean_step={clean_step}")
                    accuracy = calculate_condition_accuracy_for_deepcache_experiment(cache_clean_dir)
                    
                    results.append({
                        'experiment_dir': os.path.basename(exp_dir),
                        'cache_interval': cache_interval,
                        'clean_step': clean_step,
                        'condition_accuracy': accuracy
                    })
                    
                except (ValueError, IndexError) as e:
                    print(f"Error parsing directory name {item}: {e}")
                    continue
    
    return results

def save_results_to_csv(results, filename):
    """Save results to CSV file"""
    if not results:
        print("No results to save")
        return
    
    headers = ['experiment_dir', 'cache_interval', 'clean_step', 'condition_accuracy']
    
    with open(filename, 'w') as f:
        # Write header
        f.write(','.join(headers) + '\n')
        
        # Write data
        for result in results:
            if result['condition_accuracy'] is not None:
                f.write(f"{result['experiment_dir']},{result['cache_interval']},{result['clean_step']},{result['condition_accuracy']:.4f}\n")
    
    print(f"Results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Calculate condition accuracy for DeepCache experiments')
    parser.add_argument('--experiment_dir', type=str, default='.', 
                       help='Base directory containing DeepCache experiment directories')
    parser.add_argument('--target_class', type=int, default=2,
                       help='Target class for condition accuracy calculation (default: 2 for bird class)')
    parser.add_argument('--output_file', type=str, default='deepcache_condition_accuracy_results.csv',
                       help='Output CSV file name')
    
    args = parser.parse_args()
    
    # DeepCache 실험 처리
    results = process_deepcache_experiments(args.experiment_dir)
    
    if results:
        # CSV 파일로 저장
        save_results_to_csv(results, args.output_file)
        
        # 결과 요약
        print(f"\n=== Summary ===")
        print(f"Total experiments processed: {len(results)}")
        valid_results = [r for r in results if r['condition_accuracy'] is not None]
        print(f"Valid results: {len(valid_results)}")
        
        if valid_results:
            accuracies = [r['condition_accuracy'] for r in valid_results]
            print(f"Average condition accuracy: {np.mean(accuracies):.4f}")
            print(f"Min condition accuracy: {np.min(accuracies):.4f}")
            print(f"Max condition accuracy: {np.max(accuracies):.4f}")
            
            # Cache interval별 평균
            cache_intervals = set(r['cache_interval'] for r in valid_results)
            print(f"\n=== Cache Interval별 평균 ===")
            for cache_interval in sorted(cache_intervals):
                interval_results = [r for r in valid_results if r['cache_interval'] == cache_interval]
                interval_accuracies = [r['condition_accuracy'] for r in interval_results]
                print(f"Cache Interval {cache_interval}: {np.mean(interval_accuracies):.4f} (n={len(interval_results)})")
            
            # Clean step별 평균
            clean_steps = set(r['clean_step'] for r in valid_results)
            print(f"\n=== Clean Step별 평균 ===")
            for clean_step in sorted(clean_steps):
                step_results = [r for r in valid_results if r['clean_step'] == clean_step]
                step_accuracies = [r['condition_accuracy'] for r in step_results]
                print(f"Clean Step {clean_step}: {np.mean(step_accuracies):.4f} (n={len(step_results)})")
    else:
        print("No results found")

if __name__ == "__main__":
    main() 