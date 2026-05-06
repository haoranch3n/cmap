"""Configuration for the DINOv2 cell-embedding pipeline.

A single frozen dataclass holds all defaults so the CLI in
``features/extract_dinov2_embeddings.py`` only has to forward overrides.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DinoV2Config:
    """Parameters controlling DINOv2 feature extraction.

    Attributes:
        model_name: torch.hub model name; ViT-B/14 returns a 768-D CLS token.
        target_size: Spatial size each z-slice is resized to before the model.
            Must be a multiple of 14 for ViT-B/14.
        norm_p_low: Lower percentile for per-channel clipping (microscopy norm).
        norm_p_high: Upper percentile for per-channel clipping.
        use_imagenet_stats: If True, apply ImageNet mean/std after [0,1] scaling.
            Off by default since this is microscopy, not natural images.
        apply_mask: If True, zero out pixels outside ``Primary_Cell_Mask``
            before normalization (suppresses neighbour cells).
        empty_slice_nonzero_frac: Skip a z-slice if the fraction of nonzero
            pixels in channel 0 (642) is below this threshold.
        batch_size: Slices encoded per forward pass.
        device: Torch device string (``"cuda"`` / ``"cpu"`` / ``"cuda:0"`` ...).
        save_slice_embeddings: If True, also dump a per-slice ``.npz`` for
            debugging / future attention-pooling experiments.
        extraction_mode: ``"z_mean"`` (mean-pool all z-slices) or
            ``"orthogonal_concat"`` (three DINOv2 views through a robust centroid,
            concatenated to ``3 * embed_dim``).
        norm_scope: ``"slice"`` uses per-slice percentiles; ``"volume"`` uses
            per-channel ``lo``/``hi`` from ``dinov2_volume_norm_bounds.csv`` next
            to the sample output (see ``volume_norm_csv`` module).
        centroid_core_quantile: DT quantile for interior core (orthogonal mode).
        centroid_use_intensity_weights: Weight core COM by summed intensities.
    """

    model_name: str = "dinov2_vitb14"
    target_size: int = 224
    norm_p_low: float = 1.0
    norm_p_high: float = 99.0
    use_imagenet_stats: bool = False
    apply_mask: bool = False
    empty_slice_nonzero_frac: float = 0.01
    batch_size: int = 32
    device: str = "cuda"
    save_slice_embeddings: bool = False
    extraction_mode: str = "z_mean"
    norm_scope: str = "slice"
    centroid_core_quantile: float = 0.75
    centroid_use_intensity_weights: bool = True

    def __post_init__(self) -> None:
        if self.target_size % 14 != 0:
            raise ValueError(
                f"target_size must be a multiple of 14 for ViT-B/14, got {self.target_size}"
            )
        if self.norm_scope == "slice" and not (
            0.0 <= self.norm_p_low < self.norm_p_high <= 100.0
        ):
            raise ValueError(
                f"Need 0 <= norm_p_low ({self.norm_p_low}) < norm_p_high ({self.norm_p_high}) <= 100"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if not 0.0 <= self.empty_slice_nonzero_frac <= 1.0:
            raise ValueError(
                f"empty_slice_nonzero_frac must be in [0, 1], got {self.empty_slice_nonzero_frac}"
            )
        if self.extraction_mode not in ("z_mean", "orthogonal_concat"):
            raise ValueError(
                f"extraction_mode must be 'z_mean' or 'orthogonal_concat', got {self.extraction_mode!r}"
            )
        if self.norm_scope not in ("slice", "volume"):
            raise ValueError(f"norm_scope must be 'slice' or 'volume', got {self.norm_scope!r}")
