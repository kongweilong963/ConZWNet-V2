import random
from typing import Callable, Iterable, Optional

import torch
from tqdm import tqdm

__all__ = [
    "ArnoldScrambler",
    "calculate_exhaustive_nc_matrix",
    "discriminability_census",
]


class ArnoldScrambler:
    """Scramble binary watermark bits using the Arnold transform."""

    def __init__(self, size: int = 32, iter_num: int = 10, device="cpu"):
        self.size = size
        self.iter_num = iter_num
        self.device = device
        self.perm_idx = self._init_indices()

    def _init_indices(self) -> torch.Tensor:
        """Precompute the flattened Arnold-transform permutation indices."""
        y, x = torch.meshgrid(
            torch.arange(self.size, device=self.device),
            torch.arange(self.size, device=self.device),
            indexing="ij",
        )

        xc, yc = x.clone(), y.clone()
        for _ in range(self.iter_num):
            xn = (xc + yc) % self.size
            yn = (xc + 2 * yc) % self.size
            xc, yc = xn, yn

        return (yc * self.size + xc).reshape(-1)

    def scramble(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the precomputed permutation to a batch of watermark vectors."""
        if x.shape[1] != self.size**2:
            return x

        indices = self.perm_idx.unsqueeze(0).expand(x.shape[0], -1).to(x.device)
        return torch.gather(x, dim=1, index=indices)


def calculate_exhaustive_nc_matrix(watermarks: torch.Tensor) -> float:
    """Return the mean off-diagonal Pearson NC of all watermark pairs."""
    num_watermarks = watermarks.shape[0]
    if num_watermarks <= 1:
        return 0.0

    watermarks = watermarks.float()
    centered = watermarks - watermarks.mean(dim=1, keepdim=True)
    norms = torch.linalg.vector_norm(centered, dim=1, keepdim=True).clamp(min=1e-8)
    normalized = centered / norms
    nc_matrix = normalized @ normalized.T

    off_diagonal_sum = nc_matrix.sum() - torch.trace(nc_matrix)
    pair_count = num_watermarks * (num_watermarks - 1)
    return (off_diagonal_sum / pair_count).item()


def discriminability_census(
    model: torch.nn.Module,
    host_loader: Iterable,
    host_dataset: Iterable,
    copyright_dataset: Iterable,
    copyright_loader: Iterable,
    scrambler: ArnoldScrambler,
    device: torch.device,
    mode: str,
    forward_adapter: Callable[..., torch.Tensor],
    num_anchors: int = 10,
    evaluation_seed: int = 42,
    threshold: float = 0.5,
    anchor_preprocess: Optional[Callable] = None,
) -> float:
    """
    Evaluate host-image or copyright-image discriminability NC.

    ``mode="host"`` fixes one copyright image and varies host images.
    ``mode="copyright"`` fixes one host image and varies copyright images.
    """
    if mode not in {"host", "copyright"}:
        raise ValueError("mode must be either 'host' or 'copyright'.")

    rng = random.Random(evaluation_seed)

    if mode == "host":
        anchor_indices = rng.sample(range(len(copyright_dataset)), num_anchors)
        target_loader = host_loader
        description = "Host-image discriminability"
    else:
        anchor_indices = rng.sample(range(len(host_dataset)), num_anchors)
        target_loader = copyright_loader
        description = "Copyright-image discriminability"

    anchor_nc_values = []
    print(f"\n[*] Starting {description} evaluation with {num_anchors} anchors.")

    for run_index, anchor_index in enumerate(anchor_indices):
        watermark_batches = []

        if mode == "host":
            fixed_copyright = copyright_dataset[anchor_index].to(device).unsqueeze(0)
        else:
            host_item = host_dataset[anchor_index]
            if anchor_preprocess is not None:
                fixed_host = anchor_preprocess(host_item)
            elif isinstance(host_item, (list, tuple)):
                fixed_host = host_item[0]
            else:
                fixed_host = host_item

            if not isinstance(fixed_host, torch.Tensor):
                raise TypeError(
                    "The fixed host image must be a tensor. Provide anchor_preprocess "
                    "when host_dataset returns PIL images."
                )

            fixed_host = fixed_host.to(device).unsqueeze(0)

        progress = tqdm(
            target_loader,
            desc=f"  Anchor {run_index + 1}/{num_anchors}",
            leave=False,
        )

        for data in progress:
            if mode == "host":
                host_images = data[0] if isinstance(data, (list, tuple)) else data
                host_images = host_images.to(device)
                copyright_images = fixed_copyright.expand(
                    host_images.size(0), -1, -1, -1
                )
            else:
                copyright_images = data[0] if isinstance(data, (list, tuple)) else data
                copyright_images = copyright_images.to(device)
                host_images = fixed_host.expand(
                    copyright_images.size(0), -1, -1, -1
                )

            with torch.no_grad():
                soft_watermarks = forward_adapter(
                    model,
                    host_images,
                    copyright_images,
                )
                watermark_bits = (soft_watermarks > threshold).float()
                watermark_bits = scrambler.scramble(watermark_bits)
                watermark_batches.append(watermark_bits.cpu())

        watermark_matrix = torch.cat(watermark_batches, dim=0)
        anchor_nc = calculate_exhaustive_nc_matrix(watermark_matrix)
        anchor_nc_values.append(anchor_nc)
        print(f"    Anchor {anchor_index}: NC={anchor_nc:.6f}")

    return sum(anchor_nc_values) / len(anchor_nc_values)
