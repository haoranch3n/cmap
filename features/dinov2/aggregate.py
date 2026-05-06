"""Z-slice aggregation strategies.

Currently only mean pooling is wired into the pipeline.  Max pooling and
attention pooling can be added here without touching the rest of the module.
"""
from __future__ import annotations

import torch


def mean_pool(slice_embeds: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``(N_valid, D)`` slice embeddings into a single ``(D,)`` vector.

    Args:
        slice_embeds: Per-slice embeddings stacked along dim 0.

    Returns:
        1-D tensor of shape ``(D,)`` on the same device / dtype.

    Raises:
        ValueError: If ``slice_embeds`` is empty.
    """
    if slice_embeds.ndim != 2:
        raise ValueError(f"Expected (N, D) tensor, got shape {tuple(slice_embeds.shape)}")
    if slice_embeds.shape[0] == 0:
        raise ValueError("Cannot mean-pool zero slices; caller must filter empty cells")
    return slice_embeds.mean(dim=0)


def max_pool(slice_embeds: torch.Tensor) -> torch.Tensor:
    """Max-pool over the slice dimension. Provided for future use."""
    if slice_embeds.ndim != 2 or slice_embeds.shape[0] == 0:
        raise ValueError(f"Expected non-empty (N, D) tensor, got {tuple(slice_embeds.shape)}")
    return slice_embeds.amax(dim=0)
