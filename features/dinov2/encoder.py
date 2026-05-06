"""Frozen DINOv2 encoder wrapper.

Loading path
------------
We use the official ``facebookresearch/dinov2`` ``torch.hub`` entry::

    torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

This is the upstream-supported loading path (no ``timm`` dependency) and the
default forward pass returns the **CLS token** -- a single vector per image.
For ViT-B/14 that vector has dimension 768, exposed via ``model.embed_dim``.

The encoder is moved to the configured device, switched to ``eval`` mode and
all parameters are frozen.  ``encode_slices`` runs under
``torch.inference_mode`` for both speed and graph-free safety.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .config import DinoV2Config

logger = logging.getLogger(__name__)


class DinoV2Encoder:
    """Frozen DINOv2 ViT encoder.

    Args:
        cfg: :class:`DinoV2Config`. Only ``model_name`` and ``device`` are
            consulted here; other fields control upstream preprocessing.
        model: Optional pre-built ``nn.Module`` (used by tests to inject a
            stub without hitting ``torch.hub``).  Must expose
            ``embed_dim`` and accept ``(B, 3, H, W)``.
    """

    def __init__(self, cfg: DinoV2Config, model: nn.Module | None = None) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if model is None:
            logger.info("Loading DINOv2 model %r via torch.hub", cfg.model_name)
            model = torch.hub.load(
                "facebookresearch/dinov2",
                cfg.model_name,
                trust_repo=True,
            )
        if not hasattr(model, "embed_dim"):
            raise AttributeError(
                "DINOv2 model is expected to expose .embed_dim; got "
                f"{type(model).__name__} without it"
            )
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.embed_dim: int = int(model.embed_dim)

    @torch.inference_mode()
    def encode_slices(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode a batch of preprocessed slices into CLS-token embeddings.

        Args:
            batch: Float tensor of shape ``(B, 3, H, W)`` already
                normalized + resized by :func:`features.dinov2.transforms.prepare_batch`.

        Returns:
            ``(B, embed_dim)`` float32 tensor on the same device as the model.
            For ``dinov2_vitb14`` this is ``(B, 768)``.

        Notes:
            The default ``forward()`` of the ``torch.hub`` DINOv2 model
            returns the global CLS token (a single per-image vector).  We do
            not call ``forward_features`` -- if patch tokens are needed
            later (e.g. for attention pooling experiments), add a separate
            method here.
        """
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError(
                f"Expected (B, 3, H, W) tensor, got shape {tuple(batch.shape)}"
            )
        if batch.shape[0] == 0:
            return torch.empty((0, self.embed_dim), dtype=torch.float32, device=self.device)
        out = self.model(batch.to(self.device, non_blocking=True))
        if out.ndim != 2 or out.shape[1] != self.embed_dim:
            raise RuntimeError(
                f"Unexpected encoder output shape {tuple(out.shape)}; "
                f"expected (B, {self.embed_dim})"
            )
        return out.float()
