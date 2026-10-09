#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate host-image and copyright-image discriminability using NC."""

import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project paths and device.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR 
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from modules.ablation_core import WatermarkFullEncoder as WatermarkEncoder
from modules.distinguishability_utils import (
    ArnoldScrambler,
    calculate_exhaustive_nc_matrix,
)

def run_10_anchor_census(model, loader, ds_fixed, mode, scrambler, device, binarize_mode, threshold=0.5, seed_val=42):
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    random.seed(seed_val)
    num_runs = 10
    anchor_indices = random.sample(range(len(ds_fixed)), num_runs)
    
    all_nc_results = []
    print(f"\n[*] Starting {mode} discriminability evaluation "
          f"(mode: {binarize_mode} | anchors: {anchor_indices})")

    for i, anchor_idx in enumerate(anchor_indices):
        all_bits = []
        raw_item = ds_fixed[anchor_idx]
        anchor_raw = raw_item[0] if isinstance(raw_item, (list, tuple)) else raw_item
        
        if not isinstance(anchor_raw, torch.Tensor):
            fixed_img = transform(anchor_raw.convert('RGB')).to(device).unsqueeze(0)
        else:
            fixed_img = anchor_raw.to(device).unsqueeze(0)
        
        pbar = tqdm(loader, desc=f"   Anchor {i+1}/{num_runs}", leave=False)
        for data in pbar:
            img = data[0].to(device) if isinstance(data, (list, tuple)) else data.to(device)
            if mode == "host":
                host, cp = img, fixed_img.expand(img.size(0), -1, -1, -1)
            else:
                host, cp = fixed_img.expand(img.size(0), -1, -1, -1), img

            with torch.no_grad():
                outputs = model(host, cp)
                v_soft = outputs[1]
                v_flat = v_soft.view(v_soft.size(0), -1)
                
                # Binarize the soft watermark representation.
                if binarize_mode == "median":
                    v_thr = v_flat.median(dim=1, keepdim=True).values
                    v_bits = (v_flat > v_thr).float()
                elif binarize_mode == "mean":
                    v_mean = v_flat.mean(dim=1, keepdim=True)
                    v_bits = (v_flat > v_mean).float()
                else:
                    v_bits = (v_flat > threshold).float()
                
                v_bits = scrambler.scramble(v_bits)
                all_bits.append(v_bits.cpu())

        watermark_matrix = torch.cat(all_bits, dim=0)
        nc = calculate_exhaustive_nc_matrix(watermark_matrix)
        all_nc_results.append(nc)
        print(f"      -> Anchor {anchor_idx}: NC={nc:.5f}")

    return np.mean(all_nc_results)

def main():
    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    CONFIG = {
    'checkpoint_path': os.path.join(PROJECT_ROOT, 'results/seed_42/weights.pth'),
    'test_data_path': os.path.join(PROJECT_ROOT, 'Mini-ImageNet-Dataset/test'),
    'copyright_path': os.path.join(PROJECT_ROOT, 'copyright'),
    'dim_watermark': 1024,
    'batch_size': 96,
    'binarize_mode': 'threshold', 
    'binarize_threshold': 0.5
    }

    from utils import CopyrightDataset, HostEvalDataset, RobustnessCollate
    test_loader = DataLoader(HostEvalDataset(CONFIG['test_data_path']), batch_size=CONFIG['batch_size'], 
                              shuffle=False, collate_fn=RobustnessCollate(224))
    cp_loader = DataLoader(CopyrightDataset(CONFIG['copyright_path'], 224), batch_size=CONFIG['batch_size'], shuffle=False)
    
    test_ds = HostEvalDataset(CONFIG['test_data_path'])
    cp_ds = CopyrightDataset(CONFIG['copyright_path'], 224)

    model = WatermarkEncoder(dim_watermark=CONFIG['dim_watermark'], pretrained=False).to(DEVICE)
    
    if not os.path.exists(CONFIG['checkpoint_path']):
        print(f"\n[Error] Checkpoint not found: {CONFIG['checkpoint_path']}")
        return

    ckpt = torch.load(CONFIG['checkpoint_path'], map_location=DEVICE)
    state_dict = ckpt.get('state_dict', ckpt)
    new_state = {k.replace("module.", "").replace("encoder_q.", ""): v 
                 for k, v in state_dict.items() if "queue" not in k and "encoder_k" not in k}
    model.load_state_dict(new_state, strict=True); model.eval()

    scrambler = ArnoldScrambler(size=32, device=DEVICE)

    print(f"\n{'='*75}\nConZWNet-V2 Discriminability Evaluation\n{'='*75}")

    # Host-image discriminability.
    avg_nc_h = run_10_anchor_census(
        model, test_loader, cp_ds, "host", scrambler, DEVICE,
        binarize_mode=CONFIG['binarize_mode'],
        threshold=CONFIG.get('binarize_threshold', 0.5), 
        seed_val=SEED
    )

    # Copyright-image discriminability.
    avg_nc_c = run_10_anchor_census(
        model, cp_loader, test_ds, "copy", scrambler, DEVICE,
        binarize_mode=CONFIG['binarize_mode'],
        threshold=CONFIG.get('binarize_threshold', 0.5),
        seed_val=SEED
    )

    print(f"\n{'='*75}\nFinal Results (Mode: {CONFIG['binarize_mode']})\n{'='*75}")
    print(f"1. Host-image discriminability NC: {avg_nc_h:.5f}")
    print(f"2. Copyright-image discriminability NC: {avg_nc_c:.5f}")
    print(f"[*] Checkpoint: {CONFIG['checkpoint_path']}")
    print(f"{'='*75}\n")

if __name__ == "__main__":
    main()
