#!/usr/bin/env python3

import os
import glob
import csv
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

from utils.configs import Arguments
from logger import setup_logger
from evaluations.image import ImageEvaluator
from tasks.networks.resnet import ResNet18


def load_images_from_experiment(exp_dir: str) -> List[Image.Image]:
    """Load images from a single experiment leaf directory that contains images.npy.
    Returns list of PIL.Image.Image objects. If not found or error, returns [].
    """
    images_file = None
    for root, dirs, files in os.walk(exp_dir):
        if "images.npy" in files:
            images_file = os.path.join(root, "images.npy")
            break
    if not images_file:
        print(f"[WARN] images.npy not found under: {exp_dir}")
        return []

    try:
        arr = np.load(images_file)
        if arr.ndim != 4 or arr.shape[-1] not in (1, 3, 4):
            print(f"[WARN] Unexpected array shape {arr.shape} at {images_file}")
        images: List[Image.Image] = []
        for i in range(arr.shape[0]):
            img_np = arr[i]
            if img_np.dtype != np.uint8:
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            if img_np.shape[-1] == 1:
                img_np = np.repeat(img_np, 3, axis=-1)
            images.append(Image.fromarray(img_np))
        return images
    except Exception as e:
        print(f"[ERROR] Failed to load {images_file}: {e}")
        return []


def build_args_for_fid(target_label: int) -> Arguments:
    args = Arguments()
    args.data_type = 'image'
    args.dataset = 'cifar10'
    args.image_size = 32
    args.eval_batch_size = 32
    args.logging_dir = "logs"
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.targets = [str(target_label)]
    args.tasks = ['label_guidance']
    args.guide_networks = ['resnet_cifar10.pt']
    return args


def calculate_fid_for_exp(exp_dir: str, target_label: int) -> float | None:
    try:
        imgs = load_images_from_experiment(exp_dir)
        if not imgs:
            return None
        args = build_args_for_fid(target_label)
        setup_logger(args)
        evaluator = ImageEvaluator(args)
        # Use cached CIFAR-10 statistics per target to avoid repeated ref computation
        project_root = '/home/juhyeong/Training-Free-Guidance'
        cache_path = os.path.join(project_root, f"cifar10-fid-stats-{target_label}.pt")
        # Pass targets as list to match expected format in loader
        target_list = args.targets if isinstance(args.targets, list) else [args.targets]
        fid = evaluator._compute_fid(imgs, args.dataset, target_list, cache_path=cache_path)
        return float(fid)
    except Exception as e:
        print(f"[ERROR] FID failed for {exp_dir}: {e}")
        return None


def calculate_condition_accuracy_for_exp(exp_dir: str, target_label: int) -> float | None:
    try:
        imgs = load_images_from_experiment(exp_dir)
        if not imgs:
            return None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier = ResNet18(targets=target_label, guide_network='resnet_cifar10.pt').to(device)
        classifier.eval()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        tensors = [transform(img) for img in imgs]
        batch = torch.stack(tensors, dim=0).to(device)

        with torch.no_grad():
            outputs = classifier(batch)  # log probs for the target class
            probs = torch.exp(outputs)
            acc = probs.mean().item()
        return float(acc)
    except Exception as e:
        print(f"[ERROR] Condition accuracy failed for {exp_dir}: {e}")
        return None


def parse_config_from_subdir_name(subdir: str) -> Dict[str, Any]:
    """Parse config from names like:
    - clean_step_15_cache_depth_2
    - clean_step_0_cache_interval_1
    Returns dict with keys: clean_step (int|None), cache_depth (int|None), cache_interval (int|None)
    """
    tokens = subdir.split('_')
    clean_step = None
    cache_depth = None
    cache_interval = None
    try:
        if 'step' in tokens:
            idx = tokens.index('step')
            if idx + 1 < len(tokens):
                clean_step = int(tokens[idx + 1])
    except Exception:
        pass
    try:
        if 'depth' in tokens:
            idx = tokens.index('depth')
            if idx + 1 < len(tokens):
                cache_depth = int(tokens[idx + 1])
    except Exception:
        pass
    try:
        if 'interval' in tokens:
            idx = tokens.index('interval')
            if idx + 1 < len(tokens):
                cache_interval = int(tokens[idx + 1])
    except Exception:
        pass
    return {
        'clean_step': clean_step if clean_step is not None else -1,
        'cache_depth': cache_depth if cache_depth is not None else -1,
        'cache_interval': cache_interval if cache_interval is not None else -1,
    }


def calculate_metrics_for_target_dir(target_dir: str, target_label: int) -> List[Dict[str, Any]]:
    print(f"\n=== Target {target_label} 메트릭 계산: {os.path.basename(target_dir)} ===")
    subdirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    subdirs.sort()

    results: List[Dict[str, Any]] = []
    for sub in subdirs:
        sub_path = os.path.join(target_dir, sub)
        print(f"  처리 중: {sub}")
        fid = calculate_fid_for_exp(sub_path, target_label)
        acc = calculate_condition_accuracy_for_exp(sub_path, target_label)
        if fid is None and acc is None:
            print(f"    [WARN] 둘 다 실패: {sub}")
            continue
        cfg = parse_config_from_subdir_name(sub)
        results.append({
            'target': target_label,
            'subdir': sub,
            'clean_step': cfg['clean_step'],
            'cache_depth': cfg['cache_depth'],
            'cache_interval': cfg['cache_interval'],
            'fid_score': fid,
            'condition_accuracy': acc,
        })
    return results


def main():
    print("=== DeepCache Optimization Multi-Target: FID & Condition Accuracy 측정 ===")
    print(f"시작 시간: {datetime.now()}")

    # Base directory of experiments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/home/juhyeong/Training-Free-Guidance/deepcache_optimization_experiments_multi_target_20250829_165220')
    args_cli = parser.parse_args()

    base_dir = args_cli.base_dir
    if not os.path.isdir(base_dir):
        print(f"[ERROR] Base directory not found: {base_dir}")
        return

    target_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith('target_') and os.path.isdir(os.path.join(base_dir, d))]
    target_dirs.sort()
    print(f"발견된 타깃 디렉토리 수: {len(target_dirs)}")

    all_results: List[Dict[str, Any]] = []
    for tdir in target_dirs:
        basename = os.path.basename(tdir)
        try:
            target_label = int(basename.split('target_')[1])
        except Exception:
            print(f"[WARN] 타깃 파싱 실패: {basename}")
            continue
        res = calculate_metrics_for_target_dir(tdir, target_label)
        all_results.extend(res)

    project_root = '/home/juhyeong/Training-Free-Guidance'

    if all_results:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(project_root, f'deepcache_opt_multi_target_metrics_{ts}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['target', 'subdir', 'clean_step', 'cache_depth', 'cache_interval', 'fid_score', 'condition_accuracy']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        print(f"\n=== 결과 요약 ===")
        print(f"총 결과 수: {len(all_results)}")
        print(f"결과 파일: {csv_path}")

        # Target-wise averages
        targets = sorted(set(r['target'] for r in all_results))
        print(f"\n=== Target별 평균 ===")
        for t in targets:
            t_res = [r for r in all_results if r['target'] == t]
            fid_vals = [r['fid_score'] for r in t_res if r['fid_score'] is not None]
            acc_vals = [r['condition_accuracy'] for r in t_res if r['condition_accuracy'] is not None]
            avg_fid = (np.mean(fid_vals) if fid_vals else None)
            avg_acc = (np.mean(acc_vals) if acc_vals else None)
            fid_str = f"{avg_fid:.4f}" if avg_fid is not None else 'N/A'
            acc_str = f"{avg_acc:.4f}" if avg_acc is not None else 'N/A'
            print(f"Target {t}: FID={fid_str}, Accuracy={acc_str} (n={len(t_res)})")

        # Config-wise averages
        print(f"\n=== 설정별 평균 ===")
        cfg_keys = sorted(set((r['clean_step'], r['cache_depth'], r['cache_interval']) for r in all_results))
        for cfg in cfg_keys:
            step, depth, interval = cfg
            cfg_res = [r for r in all_results if r['clean_step'] == step and r['cache_depth'] == depth and r['cache_interval'] == interval]
            fid_vals = [r['fid_score'] for r in cfg_res if r['fid_score'] is not None]
            acc_vals = [r['condition_accuracy'] for r in cfg_res if r['condition_accuracy'] is not None]
            avg_fid = (np.mean(fid_vals) if fid_vals else None)
            avg_acc = (np.mean(acc_vals) if acc_vals else None)
            fid_str = f"{avg_fid:.4f}" if avg_fid is not None else 'N/A'
            acc_str = f"{avg_acc:.4f}" if avg_acc is not None else 'N/A'
            print(f"clean_step={step}, cache_depth={depth}, cache_interval={interval}: FID={fid_str}, Acc={acc_str} (n={len(cfg_res)})")

        # Bests
        valid_fid = [r for r in all_results if r['fid_score'] is not None]
        if valid_fid:
            best_fid = min(valid_fid, key=lambda x: x['fid_score'])
            print(f"\n=== 최고 FID ===")
            print(f"Target {best_fid['target']}, {best_fid['subdir']}: FID = {best_fid['fid_score']:.4f}")
        valid_acc = [r for r in all_results if r['condition_accuracy'] is not None]
        if valid_acc:
            best_acc = max(valid_acc, key=lambda x: x['condition_accuracy'])
            print(f"\n=== 최고 Accuracy ===")
            print(f"Target {best_acc['target']}, {best_acc['subdir']}: Accuracy = {best_acc['condition_accuracy']:.4f}")
    else:
        print("메트릭 결과가 없습니다.")

    print(f"\n완료 시간: {datetime.now()}")


if __name__ == '__main__':
    main() 