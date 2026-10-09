#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training-time perturbation module for ConZWNet-V2.

Each image is first spatially aligned and then processed by exactly one
randomly selected operator from a pool of ten perturbations.
"""

import math
import random
from io import BytesIO
from typing import List, Tuple, Union, Optional, Callable
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

__all__ = ['JointAugmentation', 'EvalAugmentation']


# ==============================================================================
#                                Helper Functions
# ==============================================================================

def get_odd_kernel_size(candidates: List[int] = None) -> int:
    """Randomly select an odd filter kernel size from the candidate list."""
    if candidates is None:
        candidates = [3, 5, 9, 11]
    k_size = random.choice(candidates)
    return k_size if k_size % 2 != 0 else k_size + 1

# ==============================================================================
#                    Ten Independent Perturbation Operators (Stage 1)
# ==============================================================================

def corner_blackout_attack(img: Image.Image) -> Image.Image:
    """1. Black out a randomly selected image corner."""
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    bw, bh = int(0.125 * w), int(0.125 * h)
    pos = random.choice(['lt', 'rt', 'lb', 'rb'])
    if pos == 'lt': arr[0:bh, 0:bw] = 0
    elif pos == 'rt': arr[0:bh, w-bw:w] = 0
    elif pos == 'lb': arr[h-bh:h, 0:bw] = 0
    elif pos == 'rb': arr[h-bh:h, w-bw:w] = 0
    return Image.fromarray(arr)

def random_crop_resize_attack(img: Image.Image) -> Image.Image:
    """2. Randomly crop 10%--75% of the image area and resize it back."""
    w, h = img.size
    i, j, h_c, w_c = transforms.RandomResizedCrop.get_params(
        img, scale=(0.1, 0.75), ratio=(3./4., 4./3.)
    )
    return TF.resized_crop(img, i, j, h_c, w_c, (h, w))

def random_rotation_attack(img: Image.Image) -> Image.Image:
    """3. Rotate the image by an angle sampled uniformly from 0 to 180 degrees."""
    angle = random.uniform(0.0, 180.0)
    return TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)

def salt_pepper_noise_attack(img: Image.Image) -> Image.Image:
    """4. Apply salt-and-pepper noise with a corruption ratio of 1%--10%."""
    arr = np.array(img, dtype=np.uint8)
    prob = random.uniform(0.01, 0.1)
    total_pixels = arr.shape[0] * arr.shape[1]
    num_modified = int(total_pixels * prob)
    if num_modified > 0:
        coords = np.random.choice(total_pixels, num_modified, replace=False)
        flat = arr.reshape(-1, 3) if arr.ndim == 3 else arr.reshape(-1)
        flat[coords[:num_modified // 2]] = 255
        flat[coords[num_modified // 2:]] = 0
    return Image.fromarray(arr)

def gaussian_noise_attack(img: Image.Image) -> Image.Image:
    """5. Add zero-mean Gaussian noise in the 0--255 pixel domain."""
    var = random.uniform(0.001, 0.1) 
    arr = np.array(img, dtype=np.float32)
    # Scale the standard deviation to the 0--255 pixel range.
    noise = np.random.normal(0, (var ** 0.5) * 255, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)

def resize_distort_attack(img: Image.Image) -> Image.Image:
    """6. Downscale the image to 50%--90% and resize it back to its original size."""
    w_orig, h_orig = img.size
    scale = random.uniform(0.5, 0.9)
    temp_size = (int(w_orig * scale), int(h_orig * scale))
    return img.resize(temp_size, Image.BILINEAR).resize((w_orig, h_orig), Image.BILINEAR)

def mean_filtering_attack(img: Image.Image) -> Image.Image:
    """7. Apply mean filtering with a randomly selected kernel size."""
    k = get_odd_kernel_size()
    return Image.fromarray(cv2.blur(np.array(img, dtype=np.uint8), (k, k)))

def gaussian_filtering_attack(img: Image.Image) -> Image.Image:
    """8. Apply Gaussian filtering with randomly sampled kernel size and sigma."""
    k = get_odd_kernel_size()
    sigma = random.uniform(0.5, 2.0)
    return Image.fromarray(cv2.GaussianBlur(np.array(img, dtype=np.uint8), (k, k), sigma))

def median_filtering_attack(img: Image.Image) -> Image.Image:
    """9. Apply median filtering with a randomly selected kernel size."""
    k = get_odd_kernel_size()
    return Image.fromarray(cv2.medianBlur(np.array(img, dtype=np.uint8), k))

def jpeg_compress_attack(img: Image.Image) -> Image.Image:
    """10. Apply JPEG compression with a quality factor sampled from 30 to 90."""
    if img.mode != 'RGB': return img
    quality = random.randint(30, 90)
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


# ==============================================================================
#                              Augmentation Classes
# ==============================================================================

class JointAugmentation:
    """
    Apply spatial alignment followed by one randomly selected perturbation.

    All perturbations operate on the unnormalized image in the 0--255 pixel
    domain. Stage 0 performs spatial alignment, and Stage 1 selects exactly
    one of the ten perturbation operators.
    """
    
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=self.NORM_MEAN, std=self.NORM_STD)
        
        # Stage 1 operator pool containing ten independent perturbations.
        self.operator_pool: List[Callable] = [
            corner_blackout_attack,      # 1
            random_crop_resize_attack,   # 2
            random_rotation_attack,      # 3
            salt_pepper_noise_attack,    # 4
            gaussian_noise_attack,       # 5
            resize_distort_attack,       # 6
            mean_filtering_attack,       # 7
            gaussian_filtering_attack,   # 8
            median_filtering_attack,     # 9
            jpeg_compress_attack         # 10
        ]

    def __call__(self, image: Image.Image) -> torch.Tensor:
        # Stage 0: spatial alignment.
        image = TF.resize(image, 256) 
        image = TF.center_crop(image, (self.img_size, self.img_size))

        # Stage 1: apply exactly one randomly selected perturbation.
        atk_func = random.choice(self.operator_pool)
        image = atk_func(image)

        # Convert the image to a tensor.
        img_tensor = self.to_tensor(image) 

        # Normalize the tensor using ImageNet statistics. ToTensor scales
        # uint8 pixel values from [0, 255] to [0, 1].
        img_norm = self.normalize(img_tensor)

        return img_norm


class EvalAugmentation:
    """
    Deterministic preprocessing for clean validation and test images.

    Each image is resized, center-cropped, converted to a tensor, and
    normalized using ImageNet statistics.
    """
    
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    def __init__(self, img_size: int = 224):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size)
        ])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=self.NORM_MEAN, std=self.NORM_STD)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = self.transform(image)
        return self.normalize(self.to_tensor(image))
