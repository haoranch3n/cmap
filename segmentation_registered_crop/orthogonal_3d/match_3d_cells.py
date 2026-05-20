"""
3D cell matching via orthogonal-view consensus.

Ported and vectorized from:
  https://github.com/murphygroup/3DCellComposer/blob/main/segmentation_3D/match_3D_cells.py
  (Haoran Chen and Robert F. Murphy, v1.5.3)

Each voxel (z, y, x) is assigned a unique "triplet index" based on the labels
it receives from the XY, XZ, and YZ 2D segmentation views:

    triplet_index = XZ_label
                  + YZ_label * (max_XZ_label + 1)
                  + XY_label * (max_XZ_label + 1) * (max_YZ_label + 1)

Voxels sharing the same non-zero triplet index belong to the same candidate 3D
cell.  The triple loop in the original is replaced by a fully-vectorized numpy
computation; only the per-cell "fill" step retains a Python loop (few cells).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _get_indices_pandas(data: np.ndarray) -> pd.Series:
    """Group flat voxel indices by value.  Returns a Series: label → (zs, ys, xs)."""
    d = data.ravel()

    def _unravel(x):
        return np.unravel_index(x.index.to_numpy(), data.shape)

    return pd.Series(d).groupby(d).apply(_unravel)


def matching_cells_3D(
    mask_XY: np.ndarray,
    mask_XZ: np.ndarray,
    mask_YZ: np.ndarray,
    minslices: int = 4,
) -> np.ndarray:
    """
    Merge three orthogonal 2D label volumes into a single 3D label volume.

    All three masks must be in **(Z, Y, X)** layout.  Background is 0.

    When only XY and YZ views are used (XZ skipped), pass ``mask_YZ`` for both
    ``mask_XZ`` and ``mask_YZ``; the YZ constraint is then applied twice which
    is equivalent to requiring XY ∩ YZ agreement.

    Args:
        mask_XY:   (Z, Y, X) int32 labels from XY (Z-slice) segmentation.
        mask_XZ:   (Z, Y, X) int32 labels from XZ segmentation (or mask_YZ).
        mask_YZ:   (Z, Y, X) int32 labels from YZ segmentation.
        minslices: Minimum number of distinct Z-slices for a cell to be kept.

    Returns:
        (Z, Y, X) int32 — contiguous cell labels starting at 1; 0 = background.
    """
    mask_XY = np.asarray(mask_XY, dtype=np.int64)
    mask_XZ = np.asarray(mask_XZ, dtype=np.int64)
    mask_YZ = np.asarray(mask_YZ, dtype=np.int64)

    X_max = int(mask_YZ.max()) + 1  # multiplier for YZ labels
    Y_max = int(mask_XZ.max()) + 1  # multiplier for XZ labels

    # Vectorized triplet encoding (replaces triple Python loop).
    # Background voxels (XY label == 0) are encoded as 0.
    seg = np.where(
        mask_XY > 0,
        mask_XZ + mask_YZ * Y_max + mask_XY * X_max * Y_max,
        np.int64(0),
    )

    # --- group voxels by triplet index ---
    triplet_coords = _get_indices_pandas(seg)
    if len(triplet_coords) <= 1:  # only background
        return np.zeros(mask_XY.shape, dtype=np.int32)
    triplet_coords = triplet_coords.iloc[1:]  # drop background (label 0)

    # Compute volumes and sort largest-first (priority for non-overlap assignment).
    volumes = triplet_coords.apply(lambda c: len(c[0]))
    sorted_triplets = volumes.sort_values(ascending=False).index

    # Pre-compute per-XY-label coordinates for the "fill" step.
    xy_label_coords = _get_indices_pandas(mask_XY)

    result = np.zeros(mask_XY.shape, dtype=np.int32)
    result_binary = np.zeros(mask_XY.shape, dtype=bool)

    cell_idx = 1
    for triplet_idx in sorted_triplets:
        coords = triplet_coords[triplet_idx]  # (zs, ys, xs) arrays
        z_coords = coords[0]

        # Skip if this candidate overlaps with already-assigned voxels.
        if result_binary[coords].any():
            result_binary[coords] = True
            continue

        # Require minimum Z-span.
        n_unique_z = len(np.unique(z_coords))
        if n_unique_z < minslices:
            result_binary[coords] = True
            continue

        # Recover the XY label (encoded in the high bits of triplet_idx).
        xy_label = int(triplet_idx // (X_max * Y_max))
        if xy_label == 0 or xy_label not in xy_label_coords.index:
            continue
        xy_coords = xy_label_coords[xy_label]  # (zs, ys, xs) for this XY label

        z_min, z_max = int(z_coords.min()), int(z_coords.max())

        # Fill: retain XY-label voxels that fall within the z_min..z_max range.
        mask_z = (xy_coords[0] >= z_min) & (xy_coords[0] <= z_max)
        fill = (xy_coords[0][mask_z], xy_coords[1][mask_z], xy_coords[2][mask_z])

        result[fill] = cell_idx
        result_binary[fill] = True
        cell_idx += 1

    return result
