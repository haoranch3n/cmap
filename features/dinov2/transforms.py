"""Per-slice preprocessing for DINOv2.

Operations are written for a single ``(3, H, W)`` slice tensor or a batch
``(B, 3, H, W)``.  Inputs are kept on CPU until :func:`prepare_batch` is
called, which moves the tensor to the encoder device once per batch.

Normalization rationale
-----------------------
Microscopy intensities span several orders of magnitude and rarely match
ImageNet statistics.  We default to per-slice, per-channel percentile
clipping (1st-99th by default) followed by rescaling to ``[0, 1]``.  This
keeps signal in a stable range across cells and slices without any natural
image colour assumptions.  ``use_imagenet_stats`` is provided for users who
explicitly want ImageNet mean/std normalization on top, but the DINOv2
``torch.hub`` model accepts inputs in ``[0, 1]`` directly so it stays off by
default.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# Standard ImageNet stats; used only when DinoV2Config.use_imagenet_stats=True.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def is_empty_slice(intensity_chw: np.ndarray, threshold: float) -> bool:
    """Return True when the 642 channel of a slice is essentially empty.

    Args:
        intensity_chw: Float array of shape ``(3, H, W)``; channel 0 is 642.
        threshold: Required nonzero fraction in channel 0 to keep the slice.
            Set to 0 to disable filtering.
    """
    if threshold <= 0.0:
        return False
    ch0 = intensity_chw[0]
    nonzero_frac = float((ch0 > 0).mean())
    return nonzero_frac < threshold


def apply_primary_mask(intensity_chw: np.ndarray, mask_yx: np.ndarray) -> np.ndarray:
    """Zero out pixels outside the primary cell mask.

    Args:
        intensity_chw: ``(3, H, W)`` float array.
        mask_yx: ``(H, W)`` boolean array; ``True`` inside the primary cell.

    Returns:
        A new ``(3, H, W)`` array (input is not modified in place).
    """
    if intensity_chw.shape[1:] != mask_yx.shape:
        raise ValueError(
            f"Mask shape {mask_yx.shape} does not match slice shape {intensity_chw.shape[1:]}"
        )
    out = intensity_chw.copy()
    out[:, ~mask_yx] = 0.0
    return out


def percentile_normalize(
    intensity_chw: np.ndarray,
    p_low: float,
    p_high: float,
) -> np.ndarray:
    """Per-channel percentile clip + rescale to ``[0, 1]`` for one slice.

    Args:
        intensity_chw: ``(C, H, W)`` float array.
        p_low: Lower percentile (e.g. 1.0).
        p_high: Upper percentile (e.g. 99.0).

    Returns:
        Float32 array of shape ``(C, H, W)`` with values in ``[0, 1]``.
        Channels whose dynamic range is degenerate (``hi <= lo``) are
        zeroed instead of producing NaNs.
    """
    out = np.empty_like(intensity_chw, dtype=np.float32)
    for c in range(intensity_chw.shape[0]):
        plane = intensity_chw[c]
        lo = float(np.percentile(plane, p_low))
        hi = float(np.percentile(plane, p_high))
        if hi <= lo:
            out[c] = 0.0
            continue
        out[c] = np.clip((plane - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)
    return out


def fixed_channel_normalize(
    intensity_chw: np.ndarray,
    lo_vec: np.ndarray,
    hi_vec: np.ndarray,
) -> np.ndarray:
    """Per-channel linear map to ``[0, 1]`` using fixed ``lo``/``hi`` per channel.

    Args:
        intensity_chw: ``(C, H, W)`` with ``C == len(lo_vec)``.
        lo_vec, hi_vec: 1-D float arrays of length ``C`` (e.g. global min and
            99.99th percentile per channel).
    """
    out = np.empty_like(intensity_chw, dtype=np.float32)
    for c in range(intensity_chw.shape[0]):
        lo = float(lo_vec[c])
        hi = float(hi_vec[c])
        if hi <= lo:
            out[c] = 0.0
            continue
        out[c] = np.clip((intensity_chw[c] - lo) / (hi - lo), 0.0, 1.0).astype(
            np.float32, copy=False
        )
    return out


def resize_slice(slice_chw: torch.Tensor, target_size: int) -> torch.Tensor:
    """Resize a ``(3, H, W)`` or ``(B, 3, H, W)`` tensor to ``target_size``.

    Uses bilinear interpolation without ``align_corners`` to match common
    vision-transformer preprocessing.
    """
    if slice_chw.ndim == 3:
        x = slice_chw.unsqueeze(0)
        squeeze = True
    elif slice_chw.ndim == 4:
        x = slice_chw
        squeeze = False
    else:
        raise ValueError(f"Expected 3-D or 4-D tensor, got {slice_chw.ndim}-D")
    x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return x.squeeze(0) if squeeze else x


def imagenet_normalize(batch_bchw: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet mean/std to a ``(B, 3, H, W)`` tensor in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=batch_bchw.device, dtype=batch_bchw.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=batch_bchw.device, dtype=batch_bchw.dtype).view(1, 3, 1, 1)
    return (batch_bchw - mean) / std


def prepare_batch(
    intensity_zchw: np.ndarray,
    mask_zhw: np.ndarray | None,
    *,
    p_low: float,
    p_high: float,
    target_size: int,
    apply_mask: bool,
    use_imagenet_stats: bool,
    empty_slice_nonzero_frac: float,
    device: torch.device | str,
    volume_lo_hi: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Build the model-ready ``(B, 3, target_size, target_size)`` tensor.

    Args:
        intensity_zchw: ``(Z, 3, Y, X)`` float array.
        mask_zhw: Optional ``(Z, Y, X)`` boolean primary-cell mask.
        p_low, p_high: Percentile bounds when ``volume_lo_hi`` is ``None``.
        volume_lo_hi: When set, ``(lo, hi)`` each float32 ``(3,)`` — fixed
            per-channel scaling (volume / CSV bounds) instead of percentiles.
        target_size: DINOv2 input spatial size (must be multiple of 14).
        apply_mask: Whether to zero pixels outside the mask.
        use_imagenet_stats: Whether to apply ImageNet mean/std post-scaling.
        empty_slice_nonzero_frac: Threshold for dropping empty slices.
        device: Target torch device for the output batch.

    Returns:
        Tuple ``(batch, valid_indices)``:
            - ``batch`` is a float32 tensor of shape
              ``(N_valid, 3, target_size, target_size)`` on ``device``.
            - ``valid_indices`` lists the original z indices that survived
              empty-slice filtering, in order.
    """
    valid_slices: list[np.ndarray] = []
    valid_indices: list[int] = []

    for z in range(intensity_zchw.shape[0]):
        plane = intensity_zchw[z]
        if is_empty_slice(plane, empty_slice_nonzero_frac):
            continue
        if apply_mask and mask_zhw is not None:
            plane = apply_primary_mask(plane, mask_zhw[z])
        if volume_lo_hi is not None:
            lo_g, hi_g = volume_lo_hi
            plane = fixed_channel_normalize(plane, lo_g, hi_g)
        else:
            plane = percentile_normalize(plane, p_low, p_high)
        valid_slices.append(plane)
        valid_indices.append(z)

    if not valid_slices:
        empty = torch.empty((0, 3, target_size, target_size), dtype=torch.float32, device=device)
        return empty, []

    stacked = np.stack(valid_slices, axis=0)
    batch = torch.from_numpy(stacked).to(device=device, dtype=torch.float32)
    batch = resize_slice(batch, target_size)
    if use_imagenet_stats:
        batch = imagenet_normalize(batch)
    return batch, valid_indices


def prepare_planes_batch(
    planes: list[np.ndarray],
    mask_planes: list[np.ndarray | None],
    *,
    p_low: float,
    p_high: float,
    target_size: int,
    apply_mask: bool,
    use_imagenet_stats: bool,
    empty_slice_nonzero_frac: float,
    device: torch.device | str,
    volume_lo_hi: tuple[np.ndarray, np.ndarray] | None = None,
) -> torch.Tensor | None:
    """Build ``(3, 3, target_size, target_size)`` from three ``(3, H, W)`` planes.

    XY / XZ / YZ views can have different ``H, W``; each is normalized and
    resized to ``target_size`` before stacking.

    Returns ``None`` if any plane is dropped as empty. Planes are ordered
    XY, XZ, YZ.
    """
    tensors: list[torch.Tensor] = []
    for plane, mp in zip(planes, mask_planes):
        if is_empty_slice(plane, empty_slice_nonzero_frac):
            return None
        if apply_mask and mp is not None:
            plane = apply_primary_mask(plane, mp)
        if volume_lo_hi is not None:
            lo_g, hi_g = volume_lo_hi
            plane = fixed_channel_normalize(plane, lo_g, hi_g)
        else:
            plane = percentile_normalize(plane, p_low, p_high)
        t = torch.from_numpy(plane).to(device=device, dtype=torch.float32)
        t = resize_slice(t, target_size)
        tensors.append(t)
    batch = torch.stack(tensors, dim=0)
    if use_imagenet_stats:
        batch = imagenet_normalize(batch)
    return batch
