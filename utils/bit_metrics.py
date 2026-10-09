import torch

__all__ = ["calculate_bcr"]


def calculate_bcr(bits1: torch.Tensor, bits2: torch.Tensor) -> float:
    """Compute the bit correct rate between two binary tensors of shape [B, D]."""
    return (bits1 == bits2).float().mean().item()