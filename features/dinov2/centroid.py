"""Robust 3D cell centroid and orthogonal plane sampling for DINOv2."""
from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


def compute_robust_centroid(
    mask_zhw: np.ndarray,
    intensity_zchw: np.ndarray,
    *,
    core_quantile: float = 0.75,
    use_intensity_weights: bool = True,
) -> tuple[float, float, float]:
    """Interior-biased centroid resistant to thin merge bridges.

    Uses the Euclidean distance transform inside the binary mask, keeps voxels
    whose distance is at or above ``core_quantile`` of distances over the
    foreground, then returns the (optionally intensity-weighted) center of mass
    of that core. Falls back to softer cores and full-mask COM if needed.

    Args:
        mask_zhw: Boolean ``(Z, Y, X)`` primary cell mask.
        intensity_zchw: Float ``(Z, 3, Y, X)`` intensity stack.
        core_quantile: First DT quantile on foreground for the interior core.
        use_intensity_weights: Weight core voxels by sum of the three channels.

    Returns:
        ``(cz, cy, cx)`` in voxel coordinates (float), order ``Z, Y, X``.
    """
    m = mask_zhw.astype(bool, copy=False)
    z, y, x = m.shape
    if not m.any():
        logger.warning("compute_robust_centroid: empty mask; using volume center")
        return (z / 2.0 - 0.5, y / 2.0 - 0.5, x / 2.0 - 0.5)

    dt = distance_transform_edt(m)
    vals = dt[m]
    if vals.size == 0:
        return (z / 2.0 - 0.5, y / 2.0 - 0.5, x / 2.0 - 0.5)

    def _weighted_com(zz: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> tuple[float, float, float]:
        if use_intensity_weights:
            pix = intensity_zchw[zz, :, yy, xx]
            w = np.maximum(pix.sum(axis=1).astype(np.float64), 1e-9)
            sw = float(w.sum())
            return (
                float((w * zz).sum() / sw),
                float((w * yy).sum() / sw),
                float((w * xx).sum() / sw),
            )
        return float(zz.mean()), float(yy.mean()), float(xx.mean())

    q_levels = [core_quantile, 0.55, 0.35, 0.15]
    seen: set[float] = set()
    for q in q_levels:
        q = float(min(max(q, 0.0), 1.0))
        if q in seen:
            continue
        seen.add(q)
        thresh = float(np.quantile(vals, q))
        core = m & (dt >= thresh)
        if int(core.sum()) < 1:
            continue
        zz, yy, xx = np.nonzero(core)
        if zz.size >= 3:
            return _weighted_com(zz, yy, xx)

    zz, yy, xx = np.nonzero(m)
    return _weighted_com(zz, yy, xx)


def slice_orthogonal_planes(
    intensity_zchw: np.ndarray,
    mask_zhw: np.ndarray | None,
    cz: float,
    cy: float,
    cx: float,
    *,
    apply_mask: bool,
) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    """Extract XY (at z), XZ (at y), YZ (at x) planes through ``(cz,cy,cx)``."""
    z, c, y, x = intensity_zchw.shape
    if c != 3:
        raise ValueError(f"Expected 3 intensity channels, got C={c}")
    iz = int(np.clip(round(cz), 0, z - 1))
    iy = int(np.clip(round(cy), 0, y - 1))
    ix = int(np.clip(round(cx), 0, x - 1))

    xy = np.ascontiguousarray(intensity_zchw[iz], dtype=np.float32)
    # (Z, 3, X) / (Z, 3, Y) -> (3, Z, X) / (3, Z, Y) for CHW preprocessing.
    xz = np.ascontiguousarray(
        np.transpose(intensity_zchw[:, :, iy, :], (1, 0, 2)), dtype=np.float32
    )
    yz = np.ascontiguousarray(
        np.transpose(intensity_zchw[:, :, :, ix], (1, 0, 2)), dtype=np.float32
    )

    masks: list[np.ndarray | None] = [None, None, None]
    if apply_mask and mask_zhw is not None:
        masks[0] = np.ascontiguousarray(mask_zhw[iz], dtype=bool)
        masks[1] = np.ascontiguousarray(mask_zhw[:, iy, :], dtype=bool)
        masks[2] = np.ascontiguousarray(mask_zhw[:, :, ix], dtype=bool)

    return [xy, xz, yz], masks
