#!/usr/bin/env python3
"""
Extract per-Z 2D TIFF planes from volumetric microscopy data.

- DAPI (or the sole channel of a single-channel volume): raw planes written as
  ``*_DAPI.tif`` for Cellpose.
- Per-volume normalization bounds: ``segmentation_norm_bounds.csv`` is written
  once into the per-volume tif_planes subdirectory and reused on subsequent
  runs (skipped when already present).  The CSV stores ``channel_index, lo, hi``
  rows computed from ``np.percentile`` at ``SEGMENTATION_GLOBAL_VOLUME_PERCENTILES``
  over all voxels of the full 3D volume.
- 4D (Z,C,Y,X) inputs: SIGNAL_CHANNELS_EXPORT channels are also extracted as
  float32 affine-normalized planes (``*_488.tif``, ``*_560.tif``, ``*_642.tif``).
"""
from __future__ import annotations

import csv
import glob
import multiprocessing as mp
import os
import traceback
from pathlib import Path
import sys

import numpy as np
import tifffile as tiff

_SEG_ROOT = Path(__file__).resolve().parents[1]
if str(_SEG_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEG_ROOT))

try:
    from config import (
        DATA_DIR,
        OUTPUT_DIR,
        SEG_PLANE_TAGS,
        SEGMENTATION_GLOBAL_VOLUME_PERCENTILES,
        SEGMENTATION_NORM_BOUNDS_CSV,
        SIGNAL_CHANNELS_EXPORT,
        SIGNAL_VOLUME_PERCENTILES,
        TIF_PLANES_DIR,
    )
except ModuleNotFoundError:
    from segmentation.config import (
        DATA_DIR,
        OUTPUT_DIR,
        SEG_PLANE_TAGS,
        SEGMENTATION_GLOBAL_VOLUME_PERCENTILES,
        SEGMENTATION_NORM_BOUNDS_CSV,
        SIGNAL_CHANNELS_EXPORT,
        SIGNAL_VOLUME_PERCENTILES,
        TIF_PLANES_DIR,
    )

try:
    from nd2reader import ND2Reader
except ImportError:
    ND2Reader = None

IMAGE_TYPE = "tif"
IMAGE_DIR = os.fspath(DATA_DIR)
OUT_ROOT = os.fspath(TIF_PLANES_DIR)
PROCESSES = 8

DAPI_CHANNEL_INDEX = 0
CHANNEL_NAMES = {DAPI_CHANNEL_INDEX: "DAPI"}
SKIP_EXISTING = True


# ---------------------------------------------------------------------------
# Normalization CSV helpers
# ---------------------------------------------------------------------------

def _volume_percentile_bounds(vol: np.ndarray, p_lo: float, p_hi: float) -> tuple[float, float]:
    """Return (lo, hi) from exact np.percentile over all voxels."""
    flat = np.asarray(vol, dtype=np.float32).ravel()
    lo, hi = float(np.percentile(flat, p_lo)), float(np.percentile(flat, p_hi))
    if hi - lo <= 1e-3:
        return float(lo), float(lo) + 1.0
    return lo, hi


def write_norm_bounds_csv(
    csv_path: Path,
    channel_bounds: list[tuple[int, float, float]],
    source: str,
) -> None:
    """Write ``channel_index, lo, hi`` rows to *csv_path* (atomic via temp file).

    Args:
        csv_path: Destination path.
        channel_bounds: List of ``(channel_index, lo, hi)`` tuples.
        source: Human-readable source label written in the comment header row.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(csv_path) + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# source", source])
        w.writerow(["channel_index", "lo", "hi"])
        for c_idx, lo, hi in channel_bounds:
            w.writerow([c_idx, f"{lo:.8g}", f"{hi:.8g}"])
    Path(tmp).replace(csv_path)


def read_norm_bounds_csv(csv_path: Path) -> list[tuple[int, float, float]]:
    """Read rows written by :func:`write_norm_bounds_csv`.

    Returns list of ``(channel_index, lo, hi)`` tuples.
    """
    rows: list[tuple[int, float, float]] = []
    with open(csv_path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#") or row[0] == "channel_index":
                continue
            rows.append((int(row[0]), float(row[1]), float(row[2])))
    return rows


def apply_volume_affine(plane: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return ((np.asarray(plane, dtype=np.float32) - lo) / (hi - lo)).astype(np.float32)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def safe_channel_name(channel_index: int) -> str:
    return CHANNEL_NAMES.get(channel_index, f"Channel{channel_index}")


def infer_seg_tag(file_stem: str) -> str:
    """Pick a plane-filename suffix from the source TIFF stem.

    Tries the known channel wavelengths first so that e.g. ``488nm_crop``
    produces ``*_488.tif`` planes.  Falls back to ``"seg"`` when no known
    wavelength is found in the name.
    """
    for tag in ("488", "560", "642"):
        if tag in file_stem:
            return tag
    if any(k in file_stem.lower() for k in ("dapi", "nuclear", "hoechst")):
        return "DAPI"
    return "seg"


def compute_out_dir(path: str, image_dir: str, out_root: str) -> Path:
    src = Path(path)
    stem = src.stem
    src_abs = src.resolve()
    image_root = Path(image_dir).resolve()
    try:
        rel_parent = src_abs.parent.relative_to(image_root)
    except Exception:
        try:
            rel_parent = src.parent.resolve().relative_to(image_root)
        except Exception:
            rel_parent = Path()
    out_dir = Path(out_root) / rel_parent
    if stem != OUTPUT_DIR.name:
        out_dir = out_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# File processors
# ---------------------------------------------------------------------------

def process_nd2_file(nd2_path: str, out_root: str, skip_existing: bool = True) -> dict:
    if ND2Reader is None:
        raise RuntimeError("nd2reader is required for ND2 files. Install with: pip install nd2reader")

    nd2_path = str(nd2_path)
    nd2_file = Path(nd2_path)
    file_stem = nd2_file.stem
    out_dir = compute_out_dir(nd2_path, IMAGE_DIR, out_root)
    summary = {
        "file": nd2_path, "out_dir": str(out_dir),
        "saved": 0, "skipped": 0,
        "saved_signal": 0, "skipped_signal": 0,
        "t_size": None, "c_size": None, "z_size": None,
    }
    p_lo, p_hi = SEGMENTATION_GLOBAL_VOLUME_PERCENTILES

    try:
        with ND2Reader(nd2_path) as images:
            sizes = images.sizes
            t_size = sizes.get("t", 1) or 1
            c_size = sizes.get("c", 1) or 1
            z_size = sizes.get("z", 1) or 1
            summary.update({"t_size": t_size, "c_size": c_size, "z_size": z_size})
            if DAPI_CHANNEL_INDEX >= c_size:
                return summary

            images.iter_axes = "z"
            images.bundle_axes = "yx"

            for t_idx in range(t_size):
                images.default_coords["t"] = t_idx

                # --- DAPI planes (raw) + norm CSV for DAPI ---
                csv_path = out_dir / SEGMENTATION_NORM_BOUNDS_CSV
                if not (skip_existing and csv_path.exists()):
                    images.default_coords["c"] = DAPI_CHANNEL_INDEX
                    dapi_planes = []
                    for z_idx in range(z_size):
                        images.default_coords["z"] = z_idx
                        dapi_planes.append(np.asarray(images[z_idx], dtype=np.float32))
                    dapi_vol = np.stack(dapi_planes, axis=0)
                    lo, hi = _volume_percentile_bounds(dapi_vol, p_lo, p_hi)
                    write_norm_bounds_csv(csv_path, [(0, lo, hi)], source=nd2_path)
                    print(
                        f"  CSV written: {csv_path.name}  "
                        f"DAPI lo={lo:.6g} hi={hi:.6g}  (p=[{p_lo},{p_hi}])"
                    )

                images.default_coords["c"] = DAPI_CHANNEL_INDEX
                chan_name = safe_channel_name(DAPI_CHANNEL_INDEX)
                for z_idx in range(z_size):
                    images.default_coords["z"] = z_idx
                    try:
                        plane = np.asarray(images[z_idx])
                        out_name = out_dir / f"{file_stem}_t{t_idx}_z{z_idx}_{chan_name}.tif"
                        if skip_existing and out_name.exists():
                            summary["skipped"] += 1
                            continue
                        tiff.imwrite(str(out_name), plane)
                        summary["saved"] += 1
                    except Exception:
                        traceback.print_exc()

                # --- Signal channels (normalized float32 planes) ---
                sv_lo, sv_hi = SIGNAL_VOLUME_PERCENTILES
                for c_idx, tag in SIGNAL_CHANNELS_EXPORT:
                    if c_idx >= c_size:
                        continue
                    images.default_coords["c"] = c_idx
                    try:
                        planes: list[np.ndarray] = []
                        for z_idx in range(z_size):
                            images.default_coords["z"] = z_idx
                            planes.append(np.asarray(images[z_idx], dtype=np.float32))
                        vol = np.stack(planes, axis=0)
                        lo_s, hi_s = _volume_percentile_bounds(vol, sv_lo, sv_hi)
                        for z_idx in range(z_size):
                            out_name = out_dir / f"{file_stem}_t{t_idx}_z{z_idx}_{tag}.tif"
                            if skip_existing and out_name.exists():
                                summary["skipped_signal"] += 1
                                continue
                            tiff.imwrite(str(out_name), apply_volume_affine(vol[z_idx], lo_s, hi_s))
                            summary["saved_signal"] += 1
                    except Exception:
                        traceback.print_exc()
    except Exception:
        traceback.print_exc()
    return summary


def process_tif_file(tif_path: str, out_root: str, skip_existing: bool = True) -> dict:
    tif_path = str(tif_path)
    tif_file = Path(tif_path)
    file_stem = tif_file.stem
    out_dir = compute_out_dir(tif_path, IMAGE_DIR, out_root)
    summary = {
        "file": tif_path, "out_dir": str(out_dir),
        "saved": 0, "skipped": 0,
        "saved_signal": 0, "skipped_signal": 0,
        "t_size": 1, "c_size": None, "z_size": None,
    }
    p_lo, p_hi = SEGMENTATION_GLOBAL_VOLUME_PERCENTILES

    try:
        arr = np.asarray(tiff.imread(tif_path))

        if arr.ndim == 3:
            # Single-channel (Z, Y, X) — the common no-deconv case.
            z_size, _, _ = arr.shape
            summary.update({"c_size": 1, "z_size": z_size})

            # Compute and save norm bounds CSV (skip if already present).
            csv_path = out_dir / SEGMENTATION_NORM_BOUNDS_CSV
            if not (skip_existing and csv_path.exists()):
                lo, hi = _volume_percentile_bounds(arr, p_lo, p_hi)
                write_norm_bounds_csv(csv_path, [(0, lo, hi)], source=tif_path)
                print(
                    f"  CSV written: {csv_path}  "
                    f"lo={lo:.6g} hi={hi:.6g}  (p=[{p_lo},{p_hi}])"
                )
            else:
                print(f"  CSV exists (skipped): {csv_path}")

            # Tag is inferred from the source filename (e.g. "488", "560", "642", "seg").
            chan_name = infer_seg_tag(file_stem)
            for z_idx in range(z_size):
                out_name = out_dir / f"{file_stem}_t0_z{z_idx}_{chan_name}.tif"
                if skip_existing and out_name.exists():
                    summary["skipped"] += 1
                    continue
                tiff.imwrite(str(out_name), arr[z_idx])
                summary["saved"] += 1
            return summary

        if arr.ndim == 4:
            # Multi-channel (Z, C, Y, X).
            z_size, c_size, _, _ = arr.shape
            summary.update({"c_size": c_size, "z_size": z_size})
            if DAPI_CHANNEL_INDEX >= c_size:
                raise ValueError(
                    f"DAPI channel index {DAPI_CHANNEL_INDEX} not in array with C={c_size}"
                )

            # Norm bounds for DAPI channel.
            csv_path = out_dir / SEGMENTATION_NORM_BOUNDS_CSV
            if not (skip_existing and csv_path.exists()):
                dapi_vol = np.asarray(arr[:, DAPI_CHANNEL_INDEX, :, :], dtype=np.float32)
                lo, hi = _volume_percentile_bounds(dapi_vol, p_lo, p_hi)
                write_norm_bounds_csv(csv_path, [(0, lo, hi)], source=tif_path)
                print(
                    f"  CSV written: {csv_path}  "
                    f"DAPI lo={lo:.6g} hi={hi:.6g}  (p=[{p_lo},{p_hi}])"
                )
            else:
                print(f"  CSV exists (skipped): {csv_path}")

            chan_name = safe_channel_name(DAPI_CHANNEL_INDEX)
            for z_idx in range(z_size):
                out_name = out_dir / f"{file_stem}_t0_z{z_idx}_{chan_name}.tif"
                if skip_existing and out_name.exists():
                    summary["skipped"] += 1
                    continue
                tiff.imwrite(str(out_name), arr[z_idx, DAPI_CHANNEL_INDEX])
                summary["saved"] += 1

            # Signal channels (normalized float32).
            sv_lo, sv_hi = SIGNAL_VOLUME_PERCENTILES
            for c_idx, tag in SIGNAL_CHANNELS_EXPORT:
                if c_idx >= c_size:
                    continue
                vol = np.asarray(arr[:, c_idx, :, :], dtype=np.float32)
                lo_s, hi_s = _volume_percentile_bounds(vol, sv_lo, sv_hi)
                for z_idx in range(z_size):
                    out_name = out_dir / f"{file_stem}_t0_z{z_idx}_{tag}.tif"
                    if skip_existing and out_name.exists():
                        summary["skipped_signal"] += 1
                        continue
                    tiff.imwrite(str(out_name), apply_volume_affine(vol[z_idx], lo_s, hi_s))
                    summary["saved_signal"] += 1
            return summary

        raise ValueError(f"Expected 3D (Z,Y,X) or 4D (Z,C,Y,X) TIFF, got shape {arr.shape}")
    except Exception:
        traceback.print_exc()
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if IMAGE_TYPE == "nd2":
        pattern = "*.nd2"
        processor = process_nd2_file
    elif IMAGE_TYPE == "tif":
        pattern = "*.tif"
        processor = process_tif_file
    else:
        raise ValueError(f"Unsupported IMAGE_TYPE: {IMAGE_TYPE}")

    os.makedirs(OUT_ROOT, exist_ok=True)
    files = glob.glob(os.path.join(IMAGE_DIR, "**", pattern), recursive=True)
    print(f"Found {len(files)} {IMAGE_TYPE.upper()} file(s) under {IMAGE_DIR}")
    if not files:
        return

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = [(path, OUT_ROOT, SKIP_EXISTING) for path in files]
    if PROCESSES <= 1:
        for arg in args:
            summary = processor(*arg)
            print(
                f"[serial] {Path(summary['file']).name}: DAPI saved={summary['saved']} "
                f"skipped={summary['skipped']}; signal saved={summary['saved_signal']} "
                f"skipped={summary['skipped_signal']} (C={summary['c_size']}, Z={summary['z_size']})"
            )
        return

    with mp.Pool(processes=PROCESSES, maxtasksperchild=1) as pool:
        for summary in pool.starmap(processor, args):
            print(
                f"[parallel] {Path(summary['file']).name}: DAPI saved={summary['saved']} "
                f"skipped={summary['skipped']}; signal saved={summary['saved_signal']} "
                f"skipped={summary['skipped_signal']} (C={summary['c_size']}, Z={summary['z_size']}) "
                f"-> {summary['out_dir']}"
            )


if __name__ == "__main__":
    main()
