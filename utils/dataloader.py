#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset and data-loading utilities for ConZWNet-V2."""

import os
from typing import List, Tuple, Optional, Callable, Any, Union
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import EvalAugmentation, JointAugmentation

# Public dataset and collation classes.
__all__ = [
    'CopyrightDataset', 
    'HostDataset', 
    'HostEvalDataset',
    'RobustnessCollate'
]

class CopyrightDataset(Dataset):
    """Load and normalize copyright images."""

    def __init__(self, root_dir: str, img_size: int = 224):
        self.root_dir = root_dir
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        self.samples = self._scan_files()

        print(f"[*] Copyright Dataset: {len(self.samples)} images loaded.")

    def _scan_files(self) -> List[Optional[str]]:
        """Scan the copyright directory for supported image files."""
        samples = []
        if not os.path.exists(self.root_dir):
            print(f"[Warn] Copyright path '{self.root_dir}' does not exist.")
            return [None]

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    samples.append(os.path.join(root, file))

        samples.sort()

        if not samples:
            return [None]
            
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and normalize one copyright image."""
        path = self.samples[idx]
        if path is None:
            return torch.randn(3, self.img_size, self.img_size)

        try:
            image = Image.open(path).convert('RGB')
            # Resize images to a fixed spatial resolution for batching.
            if image.size != (self.img_size, self.img_size):
                image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            
            img_tensor = self.to_tensor(image)
            return self.normalize(img_tensor)
        except Exception as e:
            print(f"[Error] Loading copyright image {path}: {e}")
            # Return random noise if a corrupted file cannot be loaded.
            return torch.randn(3, self.img_size, self.img_size)


class HostDataset(Dataset):
    """Load host images and generate two augmented training views."""

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = root_dir.rstrip(os.sep)
        self.transform = transform
        self.samples = []

        if not os.path.exists(self.root_dir):
            print(f"[Error] Image path '{self.root_dir}' not found.")

        self._scan_images_only()

    def _scan_images_only(self):
        """Scan the host directory for supported image files."""
        print(f"[*] Scanning host images in: {self.root_dir}")
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    self.samples.append(os.path.join(root, file))

        # Ensure deterministic sample ordering across systems.
        self.samples.sort()

        loaded = len(self.samples)
        print("-" * 60)
        print(f" Data Alignment Report (Image Only Mode)")
        print(f" > Host Images Found : {loaded}")
        print("-" * 60)

        if loaded == 0:
            raise RuntimeError("No valid images found! Check paths.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Image.Image]:
        """Return two independently augmented views for query and key branches."""
        img_path = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                view_q = self.transform(image.copy())
                view_k = self.transform(image.copy())
                return view_q, view_k
            
            return image

        except Exception as e:
            print(f"[Error] Failed loading sample {img_path}: {e}")
            # Skip a corrupted image and try the next sample.
            return self.__getitem__((idx + 1) % len(self))


class HostEvalDataset(HostDataset):
    """Load clean PIL images for evaluation."""

    def __init__(self, root_dir: str):
        super().__init__(root_dir, transform=None)

    def __getitem__(self, idx: int) -> Image.Image:
        img_path = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            return image.copy()
        except Exception:
            return self.__getitem__((idx + 1) % len(self))


class RobustnessCollate:
    """Create paired clean and attacked batches for robustness evaluation."""

    def __init__(self, img_size=224):
        self.clean_transform = EvalAugmentation(img_size=img_size)
        self.attack_transform = JointAugmentation(img_size=img_size)

    def __call__(self, batch):
        clean_imgs = []
        aug_imgs = []

        for item in batch:
            image = item

            c_img = self.clean_transform(image)
            clean_imgs.append(c_img)

            a_img = self.attack_transform(image)
            aug_imgs.append(a_img)

        return (
            torch.stack(clean_imgs),
            torch.stack(aug_imgs),
        )
