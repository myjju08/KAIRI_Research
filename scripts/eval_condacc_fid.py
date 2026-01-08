#!/usr/bin/env python
"""
Simple evaluation helper:
- Loads generated images.npy (saved by main.py) under a root directory containing subfolders per target/run.
- Computes conditional accuracy using a classifier checkpoint (expects CIFAR-10, 32x32).
- Optionally computes FID vs CIFAR-10 train using torch_fidelity (if installed).

Assumptions:
- Each run directory contains images.npy with shape [N, C, H, W] in range [-1,1] or [0,1].
- Target label is inferred from the parent folder name that includes 'target=<k>'.
"""
import argparse
import glob
import os
import re
import sys
import tempfile
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

# Ensure project root on PYTHONPATH for local evaluation utilities
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch_fidelity
except ImportError:
    torch_fidelity = None

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except ImportError:
    AutoImageProcessor = None
    AutoModelForImageClassification = None


def load_classifier(ckpt_path: str, device: torch.device):
    """
    Load classifier from checkpoint file or HuggingFace model.
    Supports:
    - Local .pt file (resnet18-compatible)
    - HuggingFace model ID (e.g., 'ahsanjavid/convnext-tiny-finetuned-cifar10')
    - Special mapping: 'resnet_cifar10.pt' -> 'ahsanjavid/convnext-tiny-finetuned-cifar10'
    """
    # Mapping for special names
    hf_mapping = {
        'resnet_cifar10.pt': 'ahsanjavid/convnext-tiny-finetuned-cifar10',
    }
    
    # Check if it's a HuggingFace model ID or mapped name
    model_id = hf_mapping.get(ckpt_path, ckpt_path)
    
    # Try HuggingFace first if transformers is available
    if AutoModelForImageClassification is not None and (
        '/' in model_id or model_id in hf_mapping.values()
    ):
        try:
            print(f"[load] Trying HuggingFace model: {model_id}")
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForImageClassification.from_pretrained(model_id)
            model.to(device)
            model.eval()
            print(f"[load] Loaded HuggingFace model: {model_id}")
            return model, processor
        except Exception as e:
            print(f"[load] HuggingFace load failed: {e}, trying local file...")
    
    # Try local file
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt_path}")
    
    # Try to build a resnet18 for CIFAR-10 and load state_dict
    model = torchvision.models.resnet18(num_classes=10)
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # Remove possible prefix (e.g., 'module.')
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_sd[nk] = v
    model.load_state_dict(new_sd, strict=False)
    model.to(device)
    model.eval()
    return model, None


def infer_target_from_path(path: Path):
    m = re.search(r"target=([0-9]+)", str(path))
    if m:
        return int(m.group(1))
    return None


def npy_to_imgs(npy_path):
    arr = np.load(npy_path)
    # Expect [N, H, W, C] or [N, C, H, W]
    if arr.ndim != 4:
        raise ValueError(f"Unexpected shape {arr.shape} in {npy_path}")
    
    # Handle [N, H, W, C] format (convert to [N, C, H, W])
    if arr.shape[-1] == 3 and arr.shape[1] != 3:
        arr = np.transpose(arr, (0, 3, 1, 2))  # [N, H, W, C] -> [N, C, H, W]
    
    # Normalize to [0, 1] if needed
    if arr.dtype == np.uint8 or arr.max() > 1.0:
        arr = arr.astype(np.float32) / 255.0
    else:
        # If range [-1,1], rescale to [0,1]
        arr = np.clip(arr, -1, 1)
        arr = (arr + 1) / 2.0
        arr = np.clip(arr, 0, 1)
    
    return arr


@torch.no_grad()
def compute_condacc(model, imgs, target, device, processor=None, batch_size=64):
    # imgs: [N, C, H, W] in [0,1] (numpy array)
    N = imgs.shape[0]
    all_preds = []
    
    # Process in batches to avoid OOM
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_imgs = imgs[start_idx:end_idx]
        
        # Handle HuggingFace models
        if processor is not None:
            # Convert numpy to PIL Images (batch-wise)
            x_list = []
            for i in range(batch_imgs.shape[0]):
                img_np = batch_imgs[i]  # [C, H, W]
                # Convert to [H, W, C] and scale to [0, 255]
                img_np = np.transpose(img_np, (1, 2, 0))  # [H, W, C]
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                # Create PIL Image directly from numpy array
                # PIL expects RGB mode for 3-channel images
                pil_img = Image.fromarray(img_np, mode='RGB')
                x_list.append(pil_img)
            
            # Process images with HuggingFace processor
            inputs = processor(x_list, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
        else:
            # Normalize as CIFAR-10 (for ResNet)
            x = torch.from_numpy(batch_imgs).float().to(device)
            mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device)[None, :, None, None]
            std = torch.tensor([0.2471, 0.2435, 0.2616], device=device)[None, :, None, None]
            x = (x - mean) / std
            logits = model(x)
        
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
    
    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0).to(device)
    targets = torch.full_like(all_preds, target)
    acc = (all_preds == targets).float().mean().item()
    return acc


def save_imgs_to_dir(imgs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # imgs in [0,1], shape [N,C,H,W]
    imgs_t = torch.from_numpy(imgs)
    grid = torchvision.utils.make_grid(imgs_t, nrow=16, padding=0)
    torchvision.utils.save_image(grid, out_dir / "grid.png")
    # save individual pngs (for FID)
    for i, img in enumerate(imgs_t):
        torchvision.utils.save_image(img, out_dir / f"{i:05d}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root dir containing run subfolders (lr_on/lr_off/target=K/...)")
    parser.add_argument("--classifier_ckpt", type=str, required=True, help="Path to CIFAR10 classifier checkpoint (resnet18-compatible).")
    parser.add_argument("--compute_fid", action="store_true", help="Compute FID vs CIFAR10 train using torch_fidelity.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing images (to avoid OOM).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf, processor = load_classifier(args.classifier_ckpt, device)

    run_dirs = [Path(p) for p in glob.glob(os.path.join(args.root, "**/images.npy"), recursive=True)]
    if not run_dirs:
        print("No images.npy found under", args.root)
        return

    results = []
    for npy_path in run_dirs:
        target = infer_target_from_path(npy_path)
        if target is None:
            print(f"Skip {npy_path}, cannot infer target=K from path")
            continue
        imgs = npy_to_imgs(npy_path)
        acc = compute_condacc(clf, imgs, target, device, processor, batch_size=args.batch_size)
        results.append((str(npy_path.parent), target, acc))
        print(f"[cond-acc] {npy_path.parent} target={target} acc={acc:.4f}")

    if args.compute_fid:
        try:
            from evaluations.utils.fid import calculate_fid
            from tasks.utils import load_image_dataset

            # group by run prefix (e.g., logs/lr_ablation/lr_on, lr_off)
            grouped = {}
            for npy_path in run_dirs:
                parent = str(Path(npy_path).parent)
                # prefix up to /target=K
                if "/target=" in parent:
                    run_prefix = parent.split("/target=")[0]
                else:
                    run_prefix = parent
                grouped.setdefault(run_prefix, []).append(npy_path)

            # Load CIFAR-10 reference (all classes)
            ref_images = load_image_dataset("cifar10", num_samples=-1, target=-1, return_tensor=False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for run_prefix, npy_list in grouped.items():
                pil_images = []
                for npy_path in npy_list:
                    imgs = npy_to_imgs(npy_path)  # [N,C,H,W] in [0,1]
                    for img in imgs:
                        arr = (img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                        pil_images.append(Image.fromarray(arr))
                if len(pil_images) == 0:
                    print(f"[fid] {run_prefix}: no images loaded; skip")
                    continue
                fid = calculate_fid(ref_images, pil_images, args.batch_size, device, cache_path=None)
                print(f"[fid] {run_prefix} vs CIFAR10 train: FID={fid:.3f}")
        except Exception as e:
            print(f"[fid] computation failed: {e}")

    # Summary
    if results:
        grouped = {}
        for path, tgt, acc in results:
            grouped.setdefault(path.split("/target=")[0], []).append(acc)
        for run, accs in grouped.items():
            print(f"[summary] run={run} mean_acc={np.mean(accs):.4f} over {len(accs)} targets")


if __name__ == "__main__":
    main()

