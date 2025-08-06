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

def load_images_from_experiment(exp_dir, num_runs=5, is_time_based=False):
    """Load images from an experiment directory"""
    all_images = []
    
    if is_time_based:
        # time_based_early_exit의 경우 exp_dir 자체가 하나의 run
        run_dir = exp_dir
        
        # images.npy 파일 찾기
        images_file = None
        for root, dirs, files in os.walk(run_dir):
            if "images.npy" in files:
                images_file = os.path.join(root, "images.npy")
                break
        
        if not images_file:
            print(f"Images file not found in: {run_dir}")
            return []
        
        print(f"Loading images from time-based run...")
        try:
            # Load numpy array
            img_array = np.load(images_file)
            print(f"Loaded array shape: {img_array.shape}")
            
            # img_array는 (num_samples, height, width, channels) 형태
            for i in range(img_array.shape[0]):
                img = Image.fromarray(img_array[i].astype(np.uint8))
                all_images.append(img)
            
            print(f"Loaded {img_array.shape[0]} images from time-based run")
            
        except Exception as e:
            print(f"Error loading {images_file}: {e}")
    else:
        # 기존 방식 (exp1, exp2 등)
        for run_idx in range(num_runs):
            run_dir = os.path.join(exp_dir, f"run_{run_idx}")
            
            # images.npy 파일 찾기
            images_file = None
            for root, dirs, files in os.walk(run_dir):
                if "images.npy" in files:
                    images_file = os.path.join(root, "images.npy")
                    break
            
            if not images_file:
                print(f"Images file not found in: {run_dir}")
                continue
            
            print(f"Loading images from run {run_idx}...")
            try:
                # Load numpy array
                img_array = np.load(images_file)
                print(f"Loaded array shape: {img_array.shape}")
                
                # img_array는 (num_samples, height, width, channels) 형태
                for i in range(img_array.shape[0]):
                    img = Image.fromarray(img_array[i].astype(np.uint8))
                    all_images.append(img)
                
                print(f"Loaded {img_array.shape[0]} images from run {run_idx}")
                
            except Exception as e:
                print(f"Error loading {images_file}: {e}")
    
    print(f"Total images loaded: {len(all_images)}")
    return all_images

def calculate_condition_accuracy_for_experiment(exp_dir, num_runs=5, target_class=1):
    """Calculate condition accuracy for a specific experiment"""
    try:
        # Load generated images from all runs
        generated_images = load_images_from_experiment(exp_dir, num_runs)
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

def process_exp1_experiments():
    """Process all exp1 experiments"""
    print("=== Processing exp1 experiments ===")
    
    start_gradients = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    guidance_scales = [0, 0.1, 1, 2, 5, 10]
    
    results = []
    
    for start_gradient in start_gradients:
        for guidance_scale in guidance_scales:
            exp_dir = f"logs/exp1/start_gradient_{start_gradient}/guidance_scale_{guidance_scale}"
            
            if not os.path.exists(exp_dir):
                print(f"Experiment directory not found: {exp_dir}")
                continue
            
            print(f"\nProcessing: start_gradient={start_gradient}, guidance_scale={guidance_scale}")
            accuracy = calculate_condition_accuracy_for_experiment(exp_dir)
            
            results.append({
                'experiment': 'exp1',
                'start_gradient': start_gradient,
                'guidance_scale': guidance_scale,
                'condition_accuracy': accuracy
            })
    
    return results

def process_exp2_experiments():
    """Process all exp2-version1 experiments"""
    print("=== Processing exp2-version1 experiments ===")
    
    early_exit_layers = [0, 1, 2, 3, 4, 5]
    
    results = []
    
    for early_exit_layer in early_exit_layers:
        exp_dir = f"logs/exp2-version1/early_exit_layer_{early_exit_layer}"
        
        if not os.path.exists(exp_dir):
            print(f"Experiment directory not found: {exp_dir}")
            continue
        
        print(f"\nProcessing: early_exit_layer={early_exit_layer}")
        accuracy = calculate_condition_accuracy_for_experiment(exp_dir)
        
        results.append({
            'experiment': 'exp2-version1',
            'early_exit_layer': early_exit_layer,
            'condition_accuracy': accuracy
        })
    
    return results

def process_time_based_early_exit_experiments():
    """Process all time_based_early_exit experiments from exp3"""
    print("=== Processing time_based_early_exit experiments from exp3 ===")
    
    # exp3 디렉토리 확인
    exp3_dir = "logs/exp3"
    if not os.path.exists(exp3_dir):
        print(f"exp3 directory not found: {exp3_dir}")
        return []
    
    results = []
    
    # 3개 설정만 처리
    target_configs = ['555555', '555550', '555000', '555500', '500000', '550000']
    
    for config_name in target_configs:
        config_dir = os.path.join(exp3_dir, config_name)
        if not os.path.isdir(config_dir):
            print(f"Config directory not found: {config_dir}")
            continue
        
        print(f"\nProcessing: exp3 {config_name}")
        
        try:
            # Load generated images from the config directory (exp3 구조에 맞게)
            generated_images = load_images_from_experiment(config_dir, num_runs=5, is_time_based=False)
            if not generated_images:
                print(f"No images found for experiment: {config_dir}")
                results.append({
                    'experiment': 'exp3_time_based_early_exit',
                    'config': config_name,
                    'condition_accuracy': None
                })
                continue
            
            # Setup configuration
            config_args = Arguments()
            config_args.targets = ['1']  # automobile class
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
            classifier = ResNet18(targets=1, guide_network='resnet_cifar10.pt')
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
            
            results.append({
                'experiment': 'exp3_time_based_early_exit',
                'config': config_name,
                'condition_accuracy': accuracy
            })
            
        except Exception as e:
            print(f"Error calculating condition accuracy for {config_name}: {e}")
            results.append({
                'experiment': 'exp3_time_based_early_exit',
                'config': config_name,
                'condition_accuracy': None
            })
    
    return results

def save_results_to_csv(results, filename):
    """Save results to CSV file"""
    if not results:
        print("No results to save")
        return
    
    # Determine CSV headers based on first result
    first_result = results[0]
    if 'start_gradient' in first_result:
        # exp1 format
        headers = ['experiment', 'start_gradient', 'guidance_scale', 'condition_accuracy']
    elif 'early_exit_layer' in first_result:
        # exp2 format
        headers = ['experiment', 'early_exit_layer', 'condition_accuracy']
    elif 'config' in first_result:
        # time_based format
        headers = ['experiment', 'config', 'condition_accuracy']
    else:
        # time_based format (old)
        headers = ['experiment', 'run_idx', 'condition_accuracy']
    
    with open(filename, 'w') as f:
        # Write header
        f.write(','.join(headers) + '\n')
        
        # Write data
        for result in results:
            if result['condition_accuracy'] is not None:
                if 'start_gradient' in result:
                    f.write(f"{result['experiment']},{result['start_gradient']},{result['guidance_scale']},{result['condition_accuracy']:.4f}\n")
                elif 'early_exit_layer' in result:
                    f.write(f"{result['experiment']},{result['early_exit_layer']},{result['condition_accuracy']:.4f}\n")
                elif 'config' in result:
                    f.write(f"{result['experiment']},{result['config']},{result['condition_accuracy']:.4f}\n")
                else:
                    f.write(f"{result['experiment']},{result['run_idx']},{result['condition_accuracy']:.4f}\n")
    
    print(f"Results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Calculate condition accuracy for all experiments')
    parser.add_argument('--exp1', action='store_true', help='Process exp1 experiments')
    parser.add_argument('--exp2', action='store_true', help='Process exp2-version1 experiments')
    parser.add_argument('--time-based', action='store_true', help='Process time_based_early_exit experiments')
    parser.add_argument('--all', action='store_true', help='Process all experiments')
    
    args = parser.parse_args()
    
    all_results = []
    
    if args.exp1 or args.all:
        exp1_results = process_exp1_experiments()
        all_results.extend(exp1_results)
        save_results_to_csv(exp1_results, 'logs/exp1_condition_accuracy_results.csv')
    
    if args.exp2 or args.all:
        exp2_results = process_exp2_experiments()
        all_results.extend(exp2_results)
        save_results_to_csv(exp2_results, 'logs/exp2_condition_accuracy_results.csv')
    
    if args.time_based or args.all:
        time_based_results = process_time_based_early_exit_experiments()
        all_results.extend(time_based_results)
        save_results_to_csv(time_based_results, 'logs/time_based_early_exit_condition_accuracy_results.csv')
    
    if args.all:
        save_results_to_csv(all_results, 'logs/all_experiments_condition_accuracy_results.csv')
    
    print(f"\n=== Summary ===")
    print(f"Total experiments processed: {len(all_results)}")
    valid_results = [r for r in all_results if r['condition_accuracy'] is not None]
    print(f"Valid results: {len(valid_results)}")
    
    if valid_results:
        accuracies = [r['condition_accuracy'] for r in valid_results]
        print(f"Average condition accuracy: {np.mean(accuracies):.4f}")
        print(f"Min condition accuracy: {np.min(accuracies):.4f}")
        print(f"Max condition accuracy: {np.max(accuracies):.4f}")

if __name__ == "__main__":
    main() 