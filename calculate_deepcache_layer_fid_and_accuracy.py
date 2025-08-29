#!/usr/bin/env python3

import os
import glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import csv
from datetime import datetime
from utils.configs import Arguments
from logger import setup_logger
from evaluations.image import ImageEvaluator
from tasks.networks.resnet import ResNet18

def load_images_from_deepcache_layer_experiment(exp_dir):
    """Load images from a DeepCache layer experiment directory"""
    all_images = []
    
    # DeepCache layer 실험 디렉토리 구조: cache_interval_X_layer_depth_Y/guidance_name=.../model=.../guide_net=.../target=.../bon=.../guidance_strength=...
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

def calculate_fid_for_deepcache_layer_experiment(exp_dir, target_label):
    """Calculate FID score for a specific DeepCache layer experiment"""
    try:
        # Load generated images
        generated_images = load_images_from_deepcache_layer_experiment(exp_dir)
        if not generated_images:
            print(f"No images found for experiment: {exp_dir}")
            return None
        
        # Setup configuration for FID calculation
        config_args = Arguments()
        config_args.data_type = 'image'
        config_args.dataset = 'cifar10'
        config_args.image_size = 32
        config_args.eval_batch_size = 32
        config_args.logging_dir = "logs"
        config_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config_args.targets = [str(target_label)]  # target class for FID calculation
        config_args.tasks = ['label_guidance']
        config_args.guide_networks = ['resnet_cifar10.pt']
        
        # Setup logger
        setup_logger(config_args)
        
        # Create ImageEvaluator for FID calculation
        evaluator = ImageEvaluator(config_args)
        
        # Calculate FID score
        print(f"Calculating FID score for {len(generated_images)} images...")
        # target을 단일 값으로 전달 (첫 번째 값 사용)
        target = config_args.targets[0] if isinstance(config_args.targets, list) else config_args.targets
        fid_score = evaluator._compute_fid(generated_images, config_args.dataset, target)
        
        print(f"FID score: {fid_score:.4f}")
        return fid_score
        
    except Exception as e:
        print(f"Error calculating FID for {exp_dir}: {e}")
        return None

def calculate_condition_accuracy_for_deepcache_layer_experiment(exp_dir, target_class):
    """Calculate condition accuracy for a specific DeepCache layer experiment"""
    try:
        # Load generated images
        generated_images = load_images_from_deepcache_layer_experiment(exp_dir)
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

def parse_layer_experiment_config(subdir):
    """Parse layer experiment configuration from subdirectory name"""
    # cache_interval_X_layer_depth_Y 형태 파싱
    if 'cache_interval_' in subdir and 'layer_depth_' in subdir:
        try:
            # cache_interval_5_layer_depth_15 -> cache_interval=5, layer_depth=15
            parts = subdir.split('_')
            cache_interval_idx = parts.index('interval') + 1
            layer_depth_idx = parts.index('depth') + 1
            
            cache_interval = int(parts[cache_interval_idx])
            layer_depth = int(parts[layer_depth_idx])
            
            return cache_interval, layer_depth
        except (ValueError, IndexError):
            print(f"Could not parse config from: {subdir}")
            return -1, -1
    else:
        return -1, -1

def calculate_metrics_for_layer_experiment(experiment_dir, target_label):
    """
    특정 layer 실험 디렉토리의 FID score와 condition accuracy를 계산합니다.
    """
    print(f"\n=== Target {target_label} Layer 실험 메트릭 계산 ===")
    
    # 실험 디렉토리 내의 서브디렉토리들 확인
    subdirs = [d for d in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, d))]
    
    results = []
    
    for subdir in subdirs:
        subdir_path = os.path.join(experiment_dir, subdir)
        
        print(f"  {subdir}: 처리 중...")
        
        # FID score 계산
        fid_score = calculate_fid_for_deepcache_layer_experiment(subdir_path, target_label)
        
        # Condition accuracy 계산
        condition_accuracy = calculate_condition_accuracy_for_deepcache_layer_experiment(subdir_path, target_label)
        
        if fid_score is None and condition_accuracy is None:
            print(f"    메트릭 계산 실패")
            continue
        
        # 실험 설정 파싱
        cache_interval, layer_depth = parse_layer_experiment_config(subdir)
        
        results.append({
            'target': target_label,
            'experiment_dir': os.path.basename(experiment_dir),
            'subdir': subdir,
            'cache_interval': cache_interval,
            'layer_depth': layer_depth,
            'fid_score': fid_score,
            'condition_accuracy': condition_accuracy
        })
    
    return results

def main():
    print("=== DeepCache Layer 실험 FID Score & Condition Accuracy 계산 ===")
    print(f"시작 시간: {datetime.now()}")
    
    # 프로젝트 루트 경로
    project_root = "/home/juhyeong/Training-Free-Guidance"
    
    # 실험 디렉토리들 찾기
    experiment_dirs = glob.glob(os.path.join(project_root, "deepcache_layer_experiments_target=*"))
    experiment_dirs.sort()
    
    print(f"발견된 실험 디렉토리 수: {len(experiment_dirs)}")
    
    all_results = []
    
    for experiment_dir in experiment_dirs:
        # target 번호 추출
        dir_name = os.path.basename(experiment_dir)
        if 'target=' in dir_name:
            target_str = dir_name.split('target=')[1].split('_')[0]
            try:
                target_label = int(target_str)
            except ValueError:
                print(f"Target 번호를 파싱할 수 없습니다: {dir_name}")
                continue
        else:
            print(f"Target 정보를 찾을 수 없습니다: {dir_name}")
            continue
        
        # 해당 실험의 메트릭 계산
        results = calculate_metrics_for_layer_experiment(experiment_dir, target_label)
        all_results.extend(results)
    
    # 결과 저장
    if all_results:
        # CSV 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(project_root, f"deepcache_layer_fid_accuracy_results_{timestamp}.csv")
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['target', 'experiment_dir', 'subdir', 'cache_interval', 'layer_depth', 
                         'fid_score', 'condition_accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)
        
        print(f"\n=== 결과 요약 ===")
        print(f"총 실험 수: {len(all_results)}")
        print(f"결과 파일: {csv_filename}")
        
        # Target별 평균 메트릭 계산
        targets = set(r['target'] for r in all_results)
        print(f"\n=== Target별 평균 메트릭 ===")
        for target in sorted(targets):
            target_results = [r for r in all_results if r['target'] == target]
            
            # FID 평균
            fid_scores = [r['fid_score'] for r in target_results if r['fid_score'] is not None]
            avg_fid = np.mean(fid_scores) if fid_scores else None
            
            # Condition accuracy 평균
            accuracies = [r['condition_accuracy'] for r in target_results if r['condition_accuracy'] is not None]
            avg_accuracy = np.mean(accuracies) if accuracies else None
            
            print(f"Target {target}: FID={avg_fid:.4f if avg_fid is not None else 'N/A'}, Accuracy={avg_accuracy:.4f if avg_accuracy is not None else 'N/A'} (n={len(target_results)})")
        
        # 설정별 평균 메트릭 계산
        print(f"\n=== 설정별 평균 메트릭 ===")
        configs = set((r['cache_interval'], r['layer_depth']) for r in all_results)
        for config in sorted(configs):
            cache_interval, layer_depth = config
            config_results = [r for r in all_results if 
                            r['cache_interval'] == cache_interval and 
                            r['layer_depth'] == layer_depth]
            
            # FID 평균
            fid_scores = [r['fid_score'] for r in config_results if r['fid_score'] is not None]
            avg_fid = np.mean(fid_scores) if fid_scores else None
            
            # Condition accuracy 평균
            accuracies = [r['condition_accuracy'] for r in config_results if r['condition_accuracy'] is not None]
            avg_accuracy = np.mean(accuracies) if accuracies else None
            
            print(f"Cache Interval {cache_interval}, Layer Depth {layer_depth}: "
                  f"FID={avg_fid:.4f if avg_fid is not None else 'N/A'}, Accuracy={avg_accuracy:.4f if avg_accuracy is not None else 'N/A'} (n={len(config_results)})")
        
        # 최고 성능 설정 찾기 (FID 기준)
        valid_fid_results = [r for r in all_results if r['fid_score'] is not None]
        if valid_fid_results:
            best_fid_result = min(valid_fid_results, key=lambda x: x['fid_score'])
            print(f"\n=== 최고 FID 성능 ===")
            print(f"Target {best_fid_result['target']}, {best_fid_result['subdir']}: FID = {best_fid_result['fid_score']:.4f}")
        
        # 최고 성능 설정 찾기 (Accuracy 기준)
        valid_accuracy_results = [r for r in all_results if r['condition_accuracy'] is not None]
        if valid_accuracy_results:
            best_accuracy_result = max(valid_accuracy_results, key=lambda x: x['condition_accuracy'])
            print(f"\n=== 최고 Accuracy 성능 ===")
            print(f"Target {best_accuracy_result['target']}, {best_accuracy_result['subdir']}: Accuracy = {best_accuracy_result['condition_accuracy']:.4f}")
        
    else:
        print("계산된 메트릭 결과가 없습니다.")
    
    print(f"\n완료 시간: {datetime.now()}")

if __name__ == "__main__":
    main()
