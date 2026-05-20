"""
Select the single "main" cell from a 3D label volume.

Strategy: choose the labeled cell whose 3D centroid is closest to the
geometric centre of the volume (Z/2, Y/2, X/2).  The registered crop was
generated around a target cell, so the target cell's centroid should be
the nearest to the crop centre.

Outputs:
  - binary mask  (Z, Y, X) uint8: 0=background, 1=main cell
  - boundary map (Z, Y, X) uint8: 1 on cell surface voxels (thick boundary)
  - JSON sidecar with selection metadata
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import tifffile
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from pipeline_config import MIN_CELL_VOLUME_3D


def select_main_cell(
    indexed_mask_tif: Path,
) -> dict:
    """
    Load the 3D indexed label volume and return selection metadata.

    The main cell is the one whose centroid (in voxel coordinates) is closest
    to the centre of the volume ``(Z/2, Y/2, X/2)``.

    Returns:
        dict with keys:
            selected_label (int | None): label chosen; None if volume is empty.
            centroid_zyx   (list[float]): centroid of the selected cell.
            volume_voxels  (int):         voxel count of the selected cell.
            n_cells_total  (int):         number of labelled cells before selection.
            n_cells_discarded (int):      cells removed.
            volume_shape   (list[int]):   (Z, Y, X) shape of the label volume.
            vol_centre_zyx (list[float]): (Z/2, Y/2, X/2) reference point.
            dist_to_centre (float):       Euclidean distance from centroid to centre.
            low_volume_warning (bool):    True if selected cell is below MIN_CELL_VOLUME_3D.
    """
    indexed_mask_tif = Path(indexed_mask_tif)
    seg_3d = tifffile.imread(str(indexed_mask_tif)).astype(np.int32)

    z, y, x = seg_3d.shape
    centre = np.array([z / 2.0, y / 2.0, x / 2.0])

    props = regionprops(seg_3d)
    if not props:
        return {
            "selected_label": None,
            "centroid_zyx": None,
            "volume_voxels": 0,
            "n_cells_total": 0,
            "n_cells_discarded": 0,
            "volume_shape": [z, y, x],
            "vol_centre_zyx": centre.tolist(),
            "dist_to_centre": None,
            "low_volume_warning": False,
        }

    best_label = None
    best_dist = float("inf")
    best_centroid = None
    best_volume = 0

    for p in props:
        centroid = np.array(p.centroid)
        dist = float(np.linalg.norm(centroid - centre))
        if dist < best_dist:
            best_dist = dist
            best_label = p.label
            best_centroid = centroid.tolist()
            best_volume = int(p.area)

    low_volume_warning = best_volume < MIN_CELL_VOLUME_3D
    if low_volume_warning:
        warnings.warn(
            f"Selected cell (label={best_label}) has only {best_volume} voxels "
            f"(< MIN_CELL_VOLUME_3D={MIN_CELL_VOLUME_3D}). "
            f"File: {indexed_mask_tif}"
        )

    return {
        "selected_label": int(best_label),
        "centroid_zyx": best_centroid,
        "volume_voxels": best_volume,
        "n_cells_total": len(props),
        "n_cells_discarded": len(props) - 1,
        "volume_shape": [z, y, x],
        "vol_centre_zyx": centre.tolist(),
        "dist_to_centre": best_dist,
        "low_volume_warning": bool(low_volume_warning),
    }


def build_main_cell_mask(indexed_mask_tif: Path, selected_label: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build binary mask and boundary map for *selected_label*.

    Returns:
        mask     (Z, Y, X) uint8 — 0=background, 1=main cell
        boundary (Z, Y, X) uint8 — 1 on cell surface (thick boundary)
    """
    seg_3d = tifffile.imread(str(indexed_mask_tif)).astype(np.int32)
    mask = (seg_3d == selected_label).astype(np.uint8)
    boundary = find_boundaries(mask.astype(bool), mode="thick", connectivity=1).astype(np.uint8)
    return mask, boundary


def run_selection(
    indexed_mask_tif: Path,
    registered_tif: Path,
    channel_name: str,
    out_dir: Path,
    skip_existing: bool = True,
) -> dict:
    """
    Full selection pipeline:
      1. Select the main cell from the 3D label volume.
      2. Build binary mask + boundary map.
      3. Assemble 5-channel output TIFF (642/488/560 original + mask + boundary).
      4. Write JSON sidecar.

    Args:
        indexed_mask_tif: Path to ``*_3D_indexed.tif``.
        registered_tif:   Path to the original ``*_registered.tif`` (Z,C,Y,X).
        channel_name:     E.g. ``"ch642"`` — used for output filenames.
        out_dir:          Directory where the 5-channel TIFF and JSON are written.
        skip_existing:    Skip if the output TIFF already exists.

    Returns:
        Selection metadata dict (from ``select_main_cell``).
    """
    indexed_mask_tif = Path(indexed_mask_tif)
    registered_tif = Path(registered_tif)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_id = registered_tif.stem.replace("_registered", "")
    out_tif = out_dir / f"{cell_id}_{channel_name}_segmented.tif"
    out_json = out_dir / f"{cell_id}_{channel_name}_selection.json"

    if skip_existing and out_tif.exists() and out_tif.stat().st_size > 0:
        meta = json.loads(out_json.read_text()) if out_json.exists() else {}
        return meta

    # Select main cell
    meta = select_main_cell(indexed_mask_tif)
    meta["cell_id"] = cell_id
    meta["channel"] = channel_name

    if meta["selected_label"] is None:
        print(f"  [warn] No cells found in {indexed_mask_tif.name}; writing empty mask.")
        arr = tifffile.imread(str(registered_tif))
        z, _, h, w = arr.shape
        empty = np.zeros((z, 5, h, w), dtype=np.float32)
        empty[:, 0] = arr[:, 0].astype(np.float32)
        empty[:, 1] = arr[:, 1].astype(np.float32)
        empty[:, 2] = arr[:, 2].astype(np.float32)
        tifffile.imwrite(str(out_tif), empty, imagej=True, metadata={"axes": "ZCYX"})
        out_json.write_text(json.dumps(meta, indent=2))
        return meta

    # Build mask + boundary
    selected_label = int(meta["selected_label"])
    binary_mask, boundary = build_main_cell_mask(indexed_mask_tif, selected_label)

    # Load original intensities (channels 0, 1, 2 = 642, 488, 560)
    arr = tifffile.imread(str(registered_tif))
    ch642 = arr[:, 0].astype(np.float32)
    ch488 = arr[:, 1].astype(np.float32)
    ch560 = arr[:, 2].astype(np.float32)

    z = ch642.shape[0]
    h, w = ch642.shape[1], ch642.shape[2]
    out_arr = np.zeros((z, 5, h, w), dtype=np.float32)
    out_arr[:, 0] = ch642
    out_arr[:, 1] = ch488
    out_arr[:, 2] = ch560
    out_arr[:, 3] = binary_mask.astype(np.float32)
    out_arr[:, 4] = boundary.astype(np.float32)

    tmp = str(out_tif) + ".tmp"
    tifffile.imwrite(tmp, out_arr, imagej=True, metadata={"axes": "ZCYX"})
    os.replace(tmp, str(out_tif))

    out_json.write_text(json.dumps(meta, indent=2))

    print(
        f"  Main cell selected: label={selected_label}, "
        f"volume={meta['volume_voxels']} vx, "
        f"dist_to_centre={meta['dist_to_centre']:.2f}, "
        f"total_cells={meta['n_cells_total']}"
    )
    return meta
