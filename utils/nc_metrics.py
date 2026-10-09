#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Normalized Correlation (NC) Metrics Module.

This module provides utility functions to calculate Pearson Correlation Coefficients
(Normalized Correlation) for evaluating watermark robustness and distinguishability.

Functions:
    - calculate_robust_nc: Pairwise correlation between original and attacked features.
    - calculate_collision_nc: Pairwise correlation matrix within a batch (self-similarity).
"""

import torch

__all__ = ['calculate_robust_nc', 'calculate_collision_nc']


def calculate_robust_nc(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Robustness NC (Normalized Correlation) between two batches of features.
    
    Mathematically equivalent to the average Pearson Correlation Coefficient.
    Used to measure similarity between original watermarks (a) and extracted watermarks (b).

    Args:
        features_a (torch.Tensor): Original features of shape (B, C).
        features_b (torch.Tensor): Attacked/Extracted features of shape (B, C).

    Returns:
        torch.Tensor: A scalar tensor representing the mean NC value across the batch.
    """
    # 1. Center the data (subtract mean)
    mean_a = features_a.mean(dim=1, keepdim=True)
    mean_b = features_b.mean(dim=1, keepdim=True)
    
    a_centered = features_a - mean_a
    b_centered = features_b - mean_b

    # 2. Compute the numerator.
    numerator = torch.sum(a_centered * b_centered, dim=1)

    # 3. Compute the product of the centered-vector norms.
    std_a = torch.sqrt(torch.sum(a_centered ** 2, dim=1))
    std_b = torch.sqrt(torch.sum(b_centered ** 2, dim=1))
    denominator = (std_a * std_b).clamp(min=1e-8)

    # 4. Calculate the per-sample NC and return the batch mean.
    nc_values = numerator / denominator
    return nc_values.mean()


def calculate_collision_nc(features: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Collision NC (Internal Correlation) within a batch.
    
    Computes the pairwise Pearson correlation matrix for all vectors in the batch
    and returns the average of the upper triangular off-diagonal elements.
    
    Used to measure distinguishability: a lower value means features are distinct.

    Args:
        features (torch.Tensor): Feature batch of shape (B, C).

    Returns:
        torch.Tensor: A scalar tensor representing the mean pairwise correlation.
                      Returns 0.0 if batch size < 2.
    """
    batch_size = features.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=features.device)

    # 1. Center the data
    mean = features.mean(dim=1, keepdim=True)
    centered = features - mean

    # 2. Normalize vectors (L2 Norm)
    # Normalizing first allows using matrix multiplication for correlation
    norm = torch.norm(centered, p=2, dim=1, keepdim=True).clamp(min=1e-8)
    normalized = centered / norm

    # 3. Compute Correlation Matrix (Cosine similarity of centered data = Pearson)
    # Shape: (B, B)
    nc_matrix = torch.mm(normalized, normalized.t())

    # 4. Extract Upper Triangular elements (excluding diagonal)
    # offset=1 excludes the diagonal (self-correlation, which is always 1)
    triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=features.device)
    
    if triu_indices.numel() == 0:
        return torch.tensor(0.0, device=features.device)

    upper_triangular_values = nc_matrix[triu_indices[0], triu_indices[1]]

    return upper_triangular_values.mean()
