#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train ConZWNet-V2 with Sigmoid binarization and MoCo-V2 InfoNCE."""

import os
import sys
import random
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project paths.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR 
for path in (CURRENT_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.append(path)

SAVE_DIR = os.path.join(PROJECT_ROOT, "results")

# Dataset paths can be overridden through environment variables.
DATASET_ROOT = os.environ.get(
    "MINI_IMAGENET_ROOT",
    os.path.join(PROJECT_ROOT, "Mini-ImageNet-Dataset"),
)
COPYRIGHT_ROOT = os.environ.get(
    "COPYRIGHT_ROOT",
    os.path.join(PROJECT_ROOT, "copyright"),
)

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from modules.ablation_core import (
        WatermarkFullEncoder,
        MoCoSinglePath,
        plot_history,
        evaluate_watermark_nc,
    )
    from utils import JointAugmentation, CopyrightDataset, HostDataset, HostEvalDataset, RobustnessCollate
except ImportError as exc:
    print(f"[CRITICAL] Import Error: {exc}")
    sys.exit(1)


# Run the script independently with seeds 42, 43, and 44.
RUN_SEED = 42

# Training configuration.
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": RUN_SEED,
    "img_size": 224,
    "batch_size": 96,
    "num_workers": 8,
    "epochs": 140,
    "lr": 0.0001,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "moco_k": 4096,
    "moco_m": 0.999,
    "moco_t": 0.07,
    "dim_watermark": 1024,
    "activation_type": "sigmoid",
    "backbone_type": "resnet50",
    "arnold_iters": 10,
    "data_path": os.path.join(DATASET_ROOT, "train"),
    "val_data_path": os.path.join(DATASET_ROOT, "val"),
    "copyright_path": COPYRIGHT_ROOT,
    "save_dir": os.path.join(SAVE_DIR, f"seed_{RUN_SEED}"),
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

def main():
    print(f"\n{'=' * 50}")
    print("ConZWNet-V2 Training")
    print(f"Output directory: {CONFIG['save_dir']}")
    print(f"{'=' * 50}\n")

    device = torch.device(CONFIG["device"]) 
    gpu_count = torch.cuda.device_count()

    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])

    # Load the training datasets.
    print("[*] Loading datasets...")
    aug = JointAugmentation(img_size=CONFIG["img_size"])
    train_ds = HostDataset(CONFIG["data_path"], transform=aug)
    cp_ds = CopyrightDataset(CONFIG["copyright_path"], img_size=CONFIG["img_size"])

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True
    )
    cp_loader = DataLoader(
        cp_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    cp_iter = iter(cp_loader)

    # Use the first 1,024 validation images for epoch-level monitoring.
    print("[*] Using the first 1,024 validation images...")
    full_val_ds = HostEvalDataset(CONFIG.get("val_data_path") or CONFIG["data_path"])
    indices = list(range(min(1024, len(full_val_ds))))
    subset_val_ds = torch.utils.data.Subset(full_val_ds, indices)

    val_loader = DataLoader(
        subset_val_ds, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=CONFIG["num_workers"],
        collate_fn=RobustnessCollate(img_size=CONFIG["img_size"]), 
        drop_last=False
    )

    # Build the MoCo-V2 model.
    print("[*] Building the MoCo-V2 model...")
    model = MoCoSinglePath(
        base_encoder=WatermarkFullEncoder,
        dim_watermark=CONFIG["dim_watermark"],
        K=CONFIG["moco_k"], m=CONFIG["moco_m"], T=CONFIG["moco_t"],
        base_encoder_kwargs={"pretrained": False},
    )
    if gpu_count > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=CONFIG["lr"], momentum=CONFIG["momentum"], weight_decay=CONFIG["weight_decay"])
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss().to(device)

    # Record loss, robustness BCR, and discriminability NC.
    loss_history, bcr_hist, dist_hist = [], [], []
    log_path = os.path.join(CONFIG["save_dir"], "training_metrics.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"{'Epoch':<6} | {'Loss':<10} | {'BCR_Robust':<12} | {'NC_Dist':<10}\n")
        f.write("-" * 52 + "\n")

    # Training loop.
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

        for _, (img_q, img_k) in pbar:
            img_q, img_k = img_q.to(device), img_k.to(device)
            try:
                I_copy = next(cp_iter)
            except StopIteration:
                cp_iter = iter(cp_loader)
                I_copy = next(cp_iter)
            I_copy = I_copy.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # The contrastive logits are computed from the normalized
                # Sigmoid output; no additional polarization loss is used.
                logits_main, _, k_norm = model(img_q, img_k, I_copy, I_copy)
                
                if isinstance(model, nn.DataParallel):
                    model.module.dequeue_and_enqueue(k_norm)
                else:
                    model.dequeue_and_enqueue(k_norm)
                
                # InfoNCE is the only training loss.
                loss = criterion(
                    logits_main,
                    torch.zeros(logits_main.shape[0], dtype=torch.long, device=device)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.3f}"})

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Epoch-level validation.
        real_encoder = model.module.encoder_q if isinstance(model, nn.DataParallel) else model.encoder_q
        
        robust_bcr, dist_nc = evaluate_watermark_nc(
            real_encoder, val_loader, cp_ds, device, CONFIG["dim_watermark"], CONFIG["arnold_iters"]
        )

        bcr_hist.append(robust_bcr)
        dist_hist.append(dist_nc)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch:<6} | {avg_loss:<10.4f} | {robust_bcr:<12.4f} | {dist_nc:<10.4f}\n")

        print(
            f"Epoch {epoch} completed. Loss: {avg_loss:.4f} | "
            f"BCR: {robust_bcr:.4f} | Dist_NC: {dist_nc:.4f}"
        )

        # Save only the checkpoint from the final epoch.
        if epoch == CONFIG["epochs"]:
            checkpoint_path = os.path.join(
                CONFIG["save_dir"],
                "weights.pth",
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[*] Final checkpoint saved to: {checkpoint_path}")

        plot_history(loss_history, CONFIG["save_dir"], "loss_curve.png", "Training Loss", "Loss")
        plot_history(bcr_hist, CONFIG["save_dir"], "robust_bcr.png", "Robustness BCR", "BCR")
        plot_history(dist_hist, CONFIG["save_dir"], "nc_coll_fixed.png", "Host Distinguishability NC", "NC")

    print(f"\n[*] Training completed. Results saved to: {CONFIG['save_dir']}")


if __name__ == "__main__":
    main()
