import os
from typing import Callable, Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from .distinguishability_utils import ArnoldScrambler


# Attack operators used by the robustness evaluation.

def rotate_attack(img: Image.Image, deg: float) -> Image.Image:
    """Rotate an image without expanding its canvas."""
    return img.rotate(deg, expand=False)

def random_crop_attack(img: Image.Image, attack_ratio: float) -> Image.Image:
    """Randomly crop the retained area and resize it to the original size."""
    w, h = img.size
    remaining_scale = 1.0 - attack_ratio

    i, j, h_c, w_c = transforms.RandomResizedCrop.get_params(
        img, 
        scale=(remaining_scale, remaining_scale), 
        ratio=(3./4., 4./3.)
    )
    
    return TF.resized_crop(img, i, j, h_c, w_c, (h, w), 
                           interpolation=transforms.InterpolationMode.BILINEAR)

def cornerCrop(img: Image.Image, pos: str, size: float) -> Image.Image:
    """Black out a fixed region in one corner of an image."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    bw, bh = int(size * w), int(size * h)
    if pos == "left_top":
        arr[0:bh, 0:bw] = 0
    elif pos == "right_top":
        arr[0:bh, w - bw : w] = 0
    elif pos == "left_bottom":
        arr[h - bh : h, 0:bw] = 0
    elif pos == "right_bottom":
        arr[h - bh : h, w - bw : w] = 0
    return Image.fromarray(arr)


def salt_pepper_noise(img: Image.Image, prob: float) -> Image.Image:
    """Apply salt-and-pepper noise."""
    arr = np.array(img)
    mask = np.random.rand(*arr.shape[:2])
    arr[mask < prob / 2] = 0
    arr[(mask >= prob / 2) & (mask < prob)] = 255
    return Image.fromarray(arr)

def gaussian_noise(img: Image.Image, var: float) -> Image.Image:
    """Apply Gaussian noise in the 0-255 pixel domain."""
    arr = np.array(img).astype(np.float32)
    sigma = (var ** 0.5) * 255
    noise = np.random.normal(0, sigma, arr.shape)
    
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

def median_filtering(img: Image.Image, k: int = 3) -> Image.Image:
    """Apply median filtering."""
    return Image.fromarray(cv2.medianBlur(np.array(img), k))


def gaussian_filtering(img: Image.Image, k: int = 3) -> Image.Image:
    """Apply Gaussian filtering."""
    return Image.fromarray(cv2.GaussianBlur(np.array(img), (k | 1, k | 1), 0))


def mean_filtering(img: Image.Image, k: int) -> Image.Image:
    """Apply mean filtering."""
    return Image.fromarray(cv2.blur(np.array(img), (k | 1, k | 1)))


def resize_attack(img: Image.Image, scale: float) -> Image.Image:
    """Downsample and restore an image to simulate resolution loss."""
    w, h = img.size
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def jpeg_compress(img: Image.Image, q: int) -> Image.Image:
    """Apply JPEG compression at the specified quality level."""
    import io
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_attacks() -> List[Tuple[str, Callable, List[float], bool]]:
    """Return attack names, operators, parameters, and geometric flags."""
    return [
        ("Rotation", rotate_attack, [10, 30, 60, 120, 150, 180], True),
        ("Random Crop", random_crop_attack, [0.25, 0.45, 0.60, 0.75], True),
        ("Corner Crop (LT)", lambda img, s: cornerCrop(img, "left_top", s), [0.125], True),
        ("Corner Crop (RT)", lambda img, s: cornerCrop(img, "right_top", s), [0.125], True),
        ("Corner Crop (LB)", lambda img, s: cornerCrop(img, "left_bottom", s), [0.125], True),
        ("Corner Crop (RB)", lambda img, s: cornerCrop(img, "right_bottom", s), [0.125], True),
        ("Salt & Pepper", salt_pepper_noise, [0.01, 0.05, 0.1], False),
        ("Gaussian Noise", gaussian_noise, [0.01, 0.05, 0.1], False),
        ("Median Filter", median_filtering, [3, 5, 9, 11], False),
        ("Gaussian Filter", gaussian_filtering, [3, 5, 9, 11], False),
        ("Mean Filter", mean_filtering, [3, 5, 9, 11], False),
        ("JPEG Compression", jpeg_compress, [80, 60, 40, 20], False),
        ("Resize Attack", resize_attack, [0.9, 0.75, 0.6, 0.5], True),
    ]


class HostTestDataset(Dataset):
    """Load host images for robustness evaluation."""
    def __init__(self, root_dir: str):
        self.root = root_dir.rstrip(os.sep)
        self.samples: List[str] = []
        self._load_samples()

    def _load_samples(self) -> List[str]:
        """Collect all supported image paths below the root directory."""
        for r, _, fs in os.walk(self.root):
            for f in fs:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    ip = os.path.join(r, f)
                    self.samples.append(ip)
        self.samples.sort()
        return self.samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ip = self.samples[idx]
        img = Image.open(ip).convert("RGB")
        return img


def pil_collate_fn(batch: Iterable):
    """Return valid PIL images without stacking them."""
    return [item for item in batch if item is not None]


def build_transforms(img_size: int = 224):
    """Build resize and ImageNet-normalization transforms."""
    norm_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    pre_process = transforms.Resize((img_size, img_size))
    return norm_tf, pre_process


def load_copyright_tensor(copyright_dir: str, norm_tf, pre_process, device: torch.device) -> torch.Tensor:
    """Load the first sorted copyright image as the fixed evaluation logo."""
    copyright_list = [f for f in os.listdir(copyright_dir) if f.lower().endswith((".png", ".jpg"))]
    if not copyright_list:
        raise FileNotFoundError("No copyright images found")
    copyright_list.sort()
    cp_tensor = norm_tf(pre_process(Image.open(os.path.join(copyright_dir, copyright_list[0])).convert("RGB"))).unsqueeze(0)
    return cp_tensor.to(device)

def build_loader(data_path: str, batch_size: int, num_workers: int, collate_fn=pil_collate_fn):
    """Build the host-image evaluation loader."""
    dataset = HostTestDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
