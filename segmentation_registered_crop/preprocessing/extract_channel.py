#!/usr/bin/env python3
"""
Extract a single fluorescence channel from a registered cell crop TIFF.

Input:  ``<cell_id>_registered.tif``  shape (Z, C, Y, X), float32
Output: one per-Z plane TIFF in ``<out_dir>/tif_planes/``
        + ``segmentation_norm_bounds.csv`` in the same folder

Normalization uses the pre-computed full-FOV bounds from
``NORM_BOUNDS_ROOT/<experiment>/<sample>_Position<N>/<ch>nm_crop/tif_planes/segmentation_norm_bounds.csv``.
Falls back to computing bounds from the crop volume if the CSV is not found.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
import yaml

_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from pipeline_config import (
    CHANNEL_NORM_FOLDER,
    NORM_BOUNDS_ROOT,
    SEGMENTATION_GLOBAL_VOLUME_PERCENTILES,
    SEGMENTATION_NORM_BOUNDS_CSV,
)


def read_norm_bounds_csv(csv_path: Path) -> tuple[float, float] | None:
    """Read channel-0 (lo, hi) from a ``segmentation_norm_bounds.csv``."""
    try:
        with open(csv_path, newline="") as fh:
            for row in csv.reader(fh):
                if not row or row[0].startswith("#") or row[0] == "channel_index":
                    continue
                if int(row[0]) == 0:
                    return float(row[1]), float(row[2])
    except Exception:
        pass
    return None


def write_norm_bounds_csv(csv_path: Path, lo: float, hi: float, source: str) -> None:
    """Write a ``segmentation_norm_bounds.csv`` compatible with the segmentation pipeline."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(csv_path) + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# source", source])
        w.writerow(["channel_index", "lo", "hi"])
        w.writerow([0, lo, hi])
    os.replace(tmp, str(csv_path))


def _volume_percentile_bounds(vol: np.ndarray, p_lo: float, p_hi: float) -> tuple[float, float]:
    flat = np.asarray(vol, dtype=np.float32).ravel()
    lo, hi = float(np.percentile(flat, p_lo)), float(np.percentile(flat, p_hi))
    if hi - lo <= 1e-3:
        return float(lo), float(lo) + 1.0
    return lo, hi


def _load_traceability(crop_dir: Path) -> dict:
    yaml_path = crop_dir / "traceability.yaml"
    if yaml_path.exists():
        with open(yaml_path) as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _lookup_full_image_norm_bounds(
    crop_dir: Path,
    channel_name: str,
) -> tuple[float, float] | None:
    """
    Derive the path to the full-FOV segmentation_norm_bounds.csv from the
    traceability.yaml and return (lo, hi), or None if not found.
    """
    meta = _load_traceability(crop_dir)
    cell_info = meta.get("cell_info", {})
    experiment = cell_info.get("experiment", "")
    sample = cell_info.get("sample", "")
    position = cell_info.get("position", "")
    if not (experiment and sample and position is not None):
        return None

    ch_folder = CHANNEL_NORM_FOLDER.get(channel_name)
    if ch_folder is None:
        return None

    csv_path = (
        NORM_BOUNDS_ROOT
        / experiment
        / f"{sample}_Position{position}"
        / ch_folder
        / "tif_planes"
        / SEGMENTATION_NORM_BOUNDS_CSV
    )
    return read_norm_bounds_csv(csv_path)


def extract_channel(
    registered_tif: Path,
    channel_idx: int,
    channel_name: str,
    out_dir: Path,
    skip_existing: bool = True,
) -> dict:
    """
    Extract one channel from a registered crop and write normalised per-Z planes.

    Args:
        registered_tif: Path to ``<cell_id>_registered.tif`` (Z, C, Y, X).
        channel_idx:    Index of the channel to extract.
        channel_name:   Human name, e.g. ``"ch642"`` — used for file naming.
        out_dir:        Per-crop per-channel working directory
                        (planes written to ``out_dir/tif_planes/``).
        skip_existing:  Skip if all planes + norm CSV already exist.

    Returns:
        dict with keys: tif_planes_dir, n_planes, lo, hi, norm_source.
    """
    registered_tif = Path(registered_tif)
    out_dir = Path(out_dir)
    tif_planes_dir = out_dir / "tif_planes"
    tif_planes_dir.mkdir(parents=True, exist_ok=True)

    cell_id = registered_tif.stem.replace("_registered", "")
    stem = f"{cell_id}_t0"

    # Check if all outputs already present.
    norm_csv = tif_planes_dir / SEGMENTATION_NORM_BOUNDS_CSV
    if skip_existing and norm_csv.exists():
        existing = list(tif_planes_dir.glob(f"{stem}_z*_{channel_name}.tif"))
        if existing:
            bounds = read_norm_bounds_csv(norm_csv)
            lo, hi = bounds if bounds else (0.0, 1.0)
            return {
                "tif_planes_dir": str(tif_planes_dir),
                "n_planes": len(existing),
                "lo": lo,
                "hi": hi,
                "norm_source": "cached",
            }

    arr = tifffile.imread(str(registered_tif))
    if arr.ndim != 4:
        raise ValueError(f"Expected (Z,C,Y,X) 4-D TIFF, got shape {arr.shape}: {registered_tif}")
    z_size, c_size, _, _ = arr.shape
    if channel_idx >= c_size:
        raise ValueError(f"channel_idx={channel_idx} out of range for C={c_size}: {registered_tif}")

    volume = np.asarray(arr[:, channel_idx, :, :], dtype=np.float32)  # (Z, Y, X)

    # Obtain normalization bounds from full-FOV CSV if available.
    norm_source = "full_fov_csv"
    bounds = _lookup_full_image_norm_bounds(registered_tif.parent, channel_name)
    if bounds is None:
        p_lo, p_hi = SEGMENTATION_GLOBAL_VOLUME_PERCENTILES
        bounds = _volume_percentile_bounds(volume, p_lo, p_hi)
        norm_source = f"crop_volume_pctile_{p_lo}_{p_hi}"
        print(
            f"  [warn] No full-FOV norm CSV for {channel_name} in {registered_tif.parent.name}; "
            f"using crop volume percentiles [{p_lo},{p_hi}]."
        )
    lo, hi = bounds

    # Affine normalize: (x - lo) / (hi - lo)
    norm_vol = (volume - lo) / (hi - lo)

    # Write per-Z planes.
    for z in range(z_size):
        plane_path = tif_planes_dir / f"{stem}_z{z}_{channel_name}.tif"
        if skip_existing and plane_path.exists() and plane_path.stat().st_size > 0:
            continue
        tifffile.imwrite(str(plane_path), norm_vol[z].astype(np.float32))

    # Write norm bounds CSV so the Cellpose 2D step can find it.
    write_norm_bounds_csv(norm_csv, lo, hi, source=str(registered_tif))
    print(
        f"  Extracted {z_size} planes for {channel_name} "
        f"(lo={lo:.4g}, hi={hi:.4g}, source={norm_source}) → {tif_planes_dir}"
    )
    return {
        "tif_planes_dir": str(tif_planes_dir),
        "n_planes": z_size,
        "lo": float(lo),
        "hi": float(hi),
        "norm_source": norm_source,
    }
