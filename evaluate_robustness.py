#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate ConZWNet-V2 robustness under 13 attack categories using BCR."""

import os
import sys
import random

# Runtime environment.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from modules.robustness_utils import (
    ArnoldScrambler,
    HostTestDataset,
    build_attacks,
    build_transforms,
    load_copyright_tensor,
    pil_collate_fn,
)
from modules.ablation_core import WatermarkFullEncoder as EncoderCls
from utils.bit_metrics import calculate_bcr


def load_model(cfg, device):
    model = EncoderCls(
        dim_watermark=cfg["dim_watermark"],
        pretrained=False,
    ).to(device)

    ckpt_path = cfg["checkpoint_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state_dict = {
        key.replace("module.", "").replace("encoder_q.", ""): value
        for key, value in state_dict.items()
        if "queue" not in key and "encoder_k" not in key
    }

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = {
    'checkpoint_path': os.path.join(PROJECT_ROOT, 'results/seed_42/weights.pth'),
    'data_path': os.path.join(PROJECT_ROOT, 'Mini-ImageNet-Dataset/test'),
    'copyright_path': os.path.join(PROJECT_ROOT, 'copyright'),
    'dim_watermark': 1024,
    'batch_size': 96,
    'arnold_iters': 10,
    'binarize_mode': 'threshold', 
    'binarize_threshold': 0.5,
    }
    try:
        model = load_model(config, device)
    except FileNotFoundError as error:
        print(f"\n[Error] {error}")
        return

    scrambler = ArnoldScrambler(
        iter_num=config["arnold_iters"],
        device=device,
    )
    norm_tf, _ = build_transforms(224)
    pre_process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    copyright_tensor = load_copyright_tensor(
        config["copyright_path"],
        norm_tf,
        pre_process,
        device,
    )

    loader = DataLoader(
        HostTestDataset(config["data_path"]),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=pil_collate_fn,
    )

    attacks = [
        (name, attack_func, params)
        for name, attack_func, params, *_ in build_attacks()
    ]

    binarize_mode = config.get("binarize_mode", "threshold")
    if binarize_mode not in ("threshold", "median", "mean"):
        raise ValueError(
            "binarize_mode must be 'threshold', 'median', or 'mean'; "
            f"received: {binarize_mode}"
        )

    print(
        f"\n{'=' * 85}\n"
        f"ConZWNet-V2 Robustness Evaluation (13 Attack Categories) "
        f"| Binarization: {binarize_mode}\n"
        f"{'=' * 85}"
    )

    for attack_name, attack_func, params in attacks:
        print(f"\n>> Category: {attack_name}")

        for param in params:
            total_bcr = 0.0
            num_images = 0
            progress = tqdm(loader, desc=f"   Param {param}", leave=False)

            for batch_images in progress:
                original_images = []
                attacked_images = []

                for image in batch_images:
                    processed_image = pre_process(image)
                    original_images.append(norm_tf(processed_image))
                    attacked_image = attack_func(processed_image, param).convert("RGB")
                    attacked_images.append(norm_tf(attacked_image))

                original_batch = torch.stack(original_images).to(device)
                attacked_batch = torch.stack(attacked_images).to(device)
                copyright_batch = copyright_tensor.expand(
                    original_batch.size(0), -1, -1, -1
                )

                with torch.no_grad():
                    original_soft = model(original_batch, copyright_batch)[1]
                    attacked_soft = model(attacked_batch, copyright_batch)[1]

                    if binarize_mode == "median":
                        original_threshold = original_soft.median(
                            dim=1, keepdim=True
                        ).values
                        attacked_threshold = attacked_soft.median(
                            dim=1, keepdim=True
                        ).values
                        original_bits = (
                            original_soft > original_threshold
                        ).float()
                        attacked_bits = (
                            attacked_soft > attacked_threshold
                        ).float()

                    elif binarize_mode == "mean":
                        original_threshold = original_soft.mean(
                            dim=1, keepdim=True
                        )
                        attacked_threshold = attacked_soft.mean(
                            dim=1, keepdim=True
                        )
                        original_bits = (
                            original_soft > original_threshold
                        ).float()
                        attacked_bits = (
                            attacked_soft > attacked_threshold
                        ).float()

                    else:
                        threshold = float(
                            config.get("binarize_threshold", 0.5)
                        )
                        original_bits = (original_soft > threshold).float()
                        attacked_bits = (attacked_soft > threshold).float()

                    original_watermark = scrambler.scramble(original_bits)
                    attacked_watermark = scrambler.scramble(attacked_bits)

                    batch_bcr = calculate_bcr(
                        original_watermark,
                        attacked_watermark,
                    )
                    total_bcr += batch_bcr * original_batch.size(0)
                    num_images += original_batch.size(0)

                if num_images >= 12000:
                    break

            if num_images > 0:
                average_bcr = total_bcr / num_images
                print(f"   [Result] Param {param!s:<6} | BCR: {average_bcr:.4f}")


if __name__ == "__main__":
    main()