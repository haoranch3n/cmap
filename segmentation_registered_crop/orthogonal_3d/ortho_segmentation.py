"""
Orthogonal (XY + YZ) 2D segmentation helper for the ortho-3d-expr experiment.

For a given normalized channel volume (Z, Y, X):
  - XY axis: segments normal Z-slices (exactly as the production pipeline).
  - YZ axis: rotates volume to (Y, Z, X), segments as if those are "Z-slices",
             then rotates the stacked mask back to (Z, Y, X).

Both axes are processed through the same multiscale Cellpose pipeline
(`run_segmentation`) and `matching_cells_2D`.  The caller then passes both
masks to `matching_cells_3D`.

Rotation math (verified):
  forward:  rot90(vol, k=1, axes=(1, 0))   (Z,Y,X) → (Y,Z,X)
  inverse:  rot90(mask, k=1, axes=(0, 1))  (Y,Z,X) → (Z,Y,X)
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import tifffile

_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from pipeline_config import (
    JI_THRESHOLD,
    SEGMENTATION_NORM_BOUNDS_CSV,
)
from multiscale_cellpose.segmentation_cellpose_2d import run_segmentation
from cellcomposor.stack_2d_planes import stack_volume
from cellcomposor.create_3d_cells import (
    bridge_gaps,
    absorb_short_fragments,
    filter_short_z_cells,
    split_disconnected_3d,
    filter_size_inconsistent,
    filter_small_volumes,
    relabel_contiguous,
)

# Re-export matching_cells_2D from the legacy path that already has the
# vectorised Hungarian-assignment implementation.
_LEGACY_SEG3D = (
    Path(__file__).resolve().parents[2]
    / "segmentation_multiscale_cellpose_3D"
    / "cellcomposor"
    / "segmentation_3D"
)
if str(_LEGACY_SEG3D) not in sys.path:
    sys.path.insert(0, str(_LEGACY_SEG3D))

from match_2D_cells import matching_cells_2D  # noqa: E402


def _read_norm_bounds(csv_path: Path) -> tuple[float, float]:
    """Read channel-0 (lo, hi) from an existing norm-bounds CSV."""
    try:
        with open(csv_path, newline="") as fh:
            for row in csv.reader(fh):
                if not row or row[0].startswith("#") or row[0] == "channel_index":
                    continue
                if int(row[0]) == 0:
                    return float(row[1]), float(row[2])
    except Exception:
        pass
    return 0.0, 1.0


def _read_normalized_volume(
    tif_planes_dir: Path,
    cell_id: str,
    channel_name: str,
    z_size: int,
) -> np.ndarray:
    """Reconstruct the (Z, Y, X) float32 normalized volume from individual Z-planes."""
    planes = []
    for z in range(z_size):
        plane_path = tif_planes_dir / f"{cell_id}_t0_z{z}_{channel_name}.tif"
        planes.append(tifffile.imread(str(plane_path)).astype(np.float32))
    return np.stack(planes, axis=0)


def _write_yz_planes(
    yz_vol: np.ndarray,
    yz_planes_dir: Path,
    cell_id: str,
    channel_name: str,
    norm_csv_src: Path,
    skip_existing: bool = True,
) -> None:
    """
    Write the (Y, Z, X) volume as per-Y plane TIFs into ``yz_planes_dir``.

    Uses the same ``_t0_z{y}_{channel}.tif`` naming convention so that
    ``run_segmentation`` and ``stack_volume`` work without modification.
    The norm-bounds CSV is copied from the XY pipeline output.
    """
    yz_planes_dir.mkdir(parents=True, exist_ok=True)
    n_y = yz_vol.shape[0]

    norm_csv_dst = yz_planes_dir / SEGMENTATION_NORM_BOUNDS_CSV
    if not norm_csv_dst.exists() and norm_csv_src.exists():
        import shutil
        shutil.copy2(str(norm_csv_src), str(norm_csv_dst))

    for y in range(n_y):
        plane_path = yz_planes_dir / f"{cell_id}_t0_z{y}_{channel_name}.tif"
        if skip_existing and plane_path.exists() and plane_path.stat().st_size > 0:
            continue
        tifffile.imwrite(str(plane_path), yz_vol[y].astype(np.float32))


def segment_xy_axis(
    tif_planes_dir: Path,
    out_dir: Path,
    gpu: bool,
    force: bool,
) -> np.ndarray | None:
    """
    Run multiscale Cellpose on XY (Z-slice) planes and apply matching_cells_2D.

    Returns the (Z, Y, X) matched label volume, or None if no masks found.
    """
    seg2d_dir = out_dir / "segmentation_2D_planes"
    diameters_dir = out_dir / "segmentation_2D_diameters"
    stacked_dir = out_dir / "segmentation_2D_stack"

    run_segmentation(
        tif_planes_root=tif_planes_dir,
        seg_root=seg2d_dir,
        diameters_root=diameters_dir,
        gpu=gpu,
    )

    stacked_path = stack_volume(seg2d_dir=seg2d_dir, stacked_dir=stacked_dir)
    if stacked_path is None:
        return None

    stack = tifffile.imread(str(stacked_path)).astype(np.int32)  # (Z, Y, X)
    print(f"  XY: stacked shape {stack.shape}, matching cells across Z …")
    matched = matching_cells_2D(stack, JI_THRESHOLD)
    return matched.astype(np.int32)


def segment_yz_axis(
    tif_planes_dir: Path,
    norm_vol: np.ndarray,
    cell_id: str,
    channel_name: str,
    out_dir: Path,
    gpu: bool,
    force: bool,
) -> np.ndarray | None:
    """
    Run multiscale Cellpose on YZ planes and apply matching_cells_2D.

    Input ``norm_vol`` is the (Z, Y, X) normalized channel volume that was
    already written to ``tif_planes_dir`` by ``extract_channel``.

    Returns the (Z, Y, X) matched label volume (rotated back), or None.
    """
    # Rotate (Z, Y, X) → (Y, Z, X); slices along Y axis, each is (Z, X).
    yz_vol = np.rot90(norm_vol, k=1, axes=(1, 0))  # (Y, Z, X)

    yz_planes_dir = out_dir / "yz_planes"
    seg2d_dir = out_dir / "yz_segmentation_2D_planes"
    diameters_dir = out_dir / "yz_segmentation_2D_diameters"
    stacked_dir = out_dir / "yz_segmentation_2D_stack"

    norm_csv_src = tif_planes_dir / SEGMENTATION_NORM_BOUNDS_CSV
    _write_yz_planes(
        yz_vol=yz_vol,
        yz_planes_dir=yz_planes_dir,
        cell_id=cell_id,
        channel_name=channel_name,
        norm_csv_src=norm_csv_src,
        skip_existing=(not force),
    )

    run_segmentation(
        tif_planes_root=yz_planes_dir,
        seg_root=seg2d_dir,
        diameters_root=diameters_dir,
        gpu=gpu,
    )

    stacked_path = stack_volume(seg2d_dir=seg2d_dir, stacked_dir=stacked_dir)
    if stacked_path is None:
        return None

    # Stacked mask is (Y, Z, X) — rotate back to (Z, Y, X).
    stack_yz = tifffile.imread(str(stacked_path)).astype(np.int32)  # (Y, Z, X)
    stack_zyx = np.rot90(stack_yz, k=1, axes=(0, 1))               # (Z, Y, X)
    print(f"  YZ: stacked shape {stack_yz.shape} → rotated {stack_zyx.shape}, matching cells …")
    matched = matching_cells_2D(stack_zyx, JI_THRESHOLD)
    return matched.astype(np.int32)


def apply_postfilters(seg_3d: np.ndarray) -> np.ndarray:
    """Apply the same quality filters as the production pipeline."""
    seg_3d = bridge_gaps(seg_3d, ji_thre=JI_THRESHOLD)
    seg_3d = absorb_short_fragments(seg_3d, max_short_span=15, ji_thre=JI_THRESHOLD)
    seg_3d = filter_short_z_cells(seg_3d)
    seg_3d = split_disconnected_3d(seg_3d)
    seg_3d = filter_size_inconsistent(seg_3d)
    seg_3d = filter_small_volumes(seg_3d)
    seg_3d = filter_short_z_cells(seg_3d)
    seg_3d = relabel_contiguous(seg_3d)
    return seg_3d
