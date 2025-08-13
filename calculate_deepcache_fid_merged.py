#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from utils.configs import Arguments
from logger import setup_logger
from evaluations.image import ImageEvaluator


def load_images_from_deepcache_experiment(exp_dir):
    """Load images from a DeepCache experiment directory"""
    all_images = []

    # Find images.npy file
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
        img_array = np.load(images_file)
        print(f"Loaded array shape: {img_array.shape}")

        # img_array is (num_samples, height, width, channels)
        for i in range(img_array.shape[0]):
            img = Image.fromarray(img_array[i].astype(np.uint8))
            all_images.append(img)

        print(f"Loaded {img_array.shape[0]} images from {exp_dir}")

    except Exception as e:
        print(f"Error loading {images_file}: {e}")
        return []

    return all_images


def calculate_fid_for_deepcache_experiment(exp_dir):
    """Calculate FID score for a specific DeepCache experiment"""
    try:
        generated_images = load_images_from_deepcache_experiment(exp_dir)
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
        # Target is not really used for FID metric, but evaluator API expects a value
        config_args.targets = ['2']
        config_args.tasks = ['label_guidance']
        config_args.guide_networks = ['resnet_cifar10.pt']

        setup_logger(config_args)

        evaluator = ImageEvaluator(config_args)

        # Compute FID score
        print(f"Calculating FID score for {len(generated_images)} images...")
        target = config_args.targets[0] if isinstance(config_args.targets, list) else config_args.targets
        fid_score = evaluator._compute_fid(generated_images, config_args.dataset, target)

        print(f"FID score: {fid_score:.4f}")
        return fid_score

    except Exception as e:
        print(f"Error calculating FID for {exp_dir}: {e}")
        return None


def process_deepcache_merged_experiments(experiment_base_dir, experiment_name):
    """Process merged DeepCache experiments that directly contain cache_X_clean_Y folders"""
    print(f"=== Processing DeepCache merged experiments from: {experiment_base_dir} ===")

    if not os.path.exists(experiment_base_dir):
        print(f"Experiment base directory not found: {experiment_base_dir}")
        return []

    results = []

    # In merged structure, cache_* folders are directly under the base directory
    for item in os.listdir(experiment_base_dir):
        if item.startswith("cache_") and os.path.isdir(os.path.join(experiment_base_dir, item)):
            cache_clean_dir = os.path.join(experiment_base_dir, item)

            # Extract X and Y from cache_X_clean_Y
            try:
                parts = item.split("_")
                cache_interval = int(parts[1])
                clean_step = int(parts[3])

                print(f"\nProcessing: cache_interval={cache_interval}, clean_step={clean_step}")
                fid_score = calculate_fid_for_deepcache_experiment(cache_clean_dir)

                results.append({
                    'experiment_dir': experiment_name,
                    'cache_interval': cache_interval,
                    'clean_step': clean_step,
                    'fid_score': fid_score
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

    headers = ['experiment_dir', 'cache_interval', 'clean_step', 'fid_score']

    with open(filename, 'w') as f:
        # Write header
        f.write(','.join(headers) + '\n')

        # Write data
        for result in results:
            if result['fid_score'] is not None:
                f.write(f"{result['experiment_dir']},{result['cache_interval']},{result['clean_step']},{result['fid_score']:.4f}\n")

    print(f"Results saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Calculate FID score for merged DeepCache experiments')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Merged experiment directory containing cache_* subfolders')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output CSV file name')

    args = parser.parse_args()

    experiment_name = os.path.basename(args.experiment_dir)

    results = process_deepcache_merged_experiments(args.experiment_dir, experiment_name)

    if results:
        save_results_to_csv(results, args.output_file)

        print(f"\n=== Summary ===")
        print(f"Total experiments processed: {len(results)}")
        valid_results = [r for r in results if r['fid_score'] is not None]
        print(f"Valid results: {len(valid_results)}")

        if valid_results:
            fid_scores = [r['fid_score'] for r in valid_results]
            print(f"Average FID score: {np.mean(fid_scores):.4f}")
            print(f"Min FID score: {np.min(fid_scores):.4f}")
            print(f"Max FID score: {np.max(fid_scores):.4f}")

            # Averages by cache interval
            cache_intervals = set(r['cache_interval'] for r in valid_results)
            print(f"\n=== Cache Interval별 평균 ===")
            for cache_interval in sorted(cache_intervals):
                interval_results = [r for r in valid_results if r['cache_interval'] == cache_interval]
                interval_fids = [r['fid_score'] for r in interval_results]
                print(f"Cache Interval {cache_interval}: {np.mean(interval_fids):.4f} (n={len(interval_results)})")

            # Averages by clean step
            clean_steps = set(r['clean_step'] for r in valid_results)
            print(f"\n=== Clean Step별 평균 ===")
            for clean_step in sorted(clean_steps):
                step_results = [r for r in valid_results if r['clean_step'] == clean_step]
                step_fids = [r['fid_score'] for r in step_results]
                print(f"Clean Step {clean_step}: {np.mean(step_fids):.4f} (n={len(step_results)})")
    else:
        print("No results found")


if __name__ == "__main__":
    main() 