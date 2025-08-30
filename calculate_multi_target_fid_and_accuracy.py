#!/usr/bin/env python3

import os
import glob
import csv
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import torch

from utils.configs import Arguments
from logger import setup_logger
from evaluations.image import ImageEvaluator


def load_images_from_experiment_dir(exp_dir):
    images_file = None
    for root, dirs, files in os.walk(exp_dir):
        if "images.npy" in files:
            images_file = os.path.join(root, "images.npy")
            break
    if not images_file:
        return []
    try:
        arr = np.load(images_file)
        images = [Image.fromarray(arr[i].astype(np.uint8)) for i in range(arr.shape[0])]
        return images
    except Exception:
        return []


def compute_metrics_for_subdir(subdir_path, target_label, device):
    images = load_images_from_experiment_dir(subdir_path)
    if not images:
        return None, None

    args = Arguments()
    args.data_type = 'image'
    args.dataset = 'cifar10'
    args.image_size = 32
    args.eval_batch_size = 64
    args.logging_dir = "logs"
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    args.targets = [str(target_label)]
    args.tasks = ['label_guidance']
    args.guide_networks = ['resnet_cifar10.pt']

    setup_logger(args)
    evaluator = ImageEvaluator(args)

    fid_score = evaluator._compute_fid(images, args.dataset, args.targets[0])

    validity, _ = evaluator._compute_validity(images, labels=args.targets)
    cond_acc = float(validity)

    return fid_score, cond_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Path to multi-target experiment root folder')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu (auto if omitted)')
    args_cli = parser.parse_args()

    root_dir = args_cli.root
    if not os.path.isdir(root_dir):
        print(f"Root dir not found: {root_dir}")
        return

    device = None
    if args_cli.device is not None:
        device = torch.device(args_cli.device)

    target_dirs = [d for d in sorted(os.listdir(root_dir)) if d.startswith('target_') and os.path.isdir(os.path.join(root_dir, d))]
    if not target_dirs:
        print("No target_* directories found.")
        return

    all_results = []

    for target_dir in target_dirs:
        target_path = os.path.join(root_dir, target_dir)
        try:
            target_label = int(target_dir.split('_')[1])
        except Exception:
            print(f"Skip directory (cannot parse target): {target_dir}")
            continue

        subdirs = [sd for sd in sorted(os.listdir(target_path)) if os.path.isdir(os.path.join(target_path, sd))]
        print(f"Target {target_label}: {len(subdirs)} subdirs")

        for sd in subdirs:
            sd_path = os.path.join(target_path, sd)
            fid, acc = compute_metrics_for_subdir(sd_path, target_label, device)
            if fid is None and acc is None:
                print(f"  {sd}: no images found or error")
                continue
            print(f"  {sd}: FID={fid:.4f if fid is not None else float('nan')}, Cond.Acc={acc:.4f if acc is not None else float('nan')}")
            all_results.append({
                'target': target_label,
                'subdir': sd,
                'fid_score': fid,
                'condition_accuracy': acc
            })

    if not all_results:
        print("No results computed.")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(root_dir, f"multi_target_fid_accuracy_{timestamp}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['target', 'subdir', 'fid_score', 'condition_accuracy'])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"\nSaved CSV: {csv_path}")

    # per-target averages
    targets = sorted(set(r['target'] for r in all_results))
    print("\n=== Per-target averages ===")
    for t in targets:
        tr = [r for r in all_results if r['target'] == t]
        fids = [r['fid_score'] for r in tr if r['fid_score'] is not None]
        accs = [r['condition_accuracy'] for r in tr if r['condition_accuracy'] is not None]
        avg_fid = np.mean(fids) if fids else None
        avg_acc = np.mean(accs) if accs else None
        print(f"Target {t}: FID={avg_fid:.4f if avg_fid is not None else 'N/A'}, Cond.Acc={avg_acc:.4f if avg_acc is not None else 'N/A'} (n={len(tr)})")


if __name__ == '__main__':
    main() 