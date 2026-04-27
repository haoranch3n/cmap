#!/usr/bin/env python3
"""
Extract per-Z 2D TIFF planes from volumetric microscopy data (DAPI only).
"""
from __future__ import annotations

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
    from config import DATA_DIR, OUTPUT_DIR, TIF_PLANES_DIR
except ModuleNotFoundError:
    from segmentation.config import DATA_DIR, OUTPUT_DIR, TIF_PLANES_DIR

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


def safe_channel_name(channel_index: int) -> str:
    return CHANNEL_NAMES.get(channel_index, f"Channel{channel_index}")


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


def process_nd2_file(nd2_path: str, out_root: str, skip_existing: bool = True) -> dict:
    if ND2Reader is None:
        raise RuntimeError("nd2reader is required for ND2 files. Install with: pip install nd2reader")

    nd2_path = str(nd2_path)
    nd2_file = Path(nd2_path)
    file_stem = nd2_file.stem
    out_dir = compute_out_dir(nd2_path, IMAGE_DIR, out_root)
    summary = {"file": nd2_path, "out_dir": str(out_dir), "saved": 0, "skipped": 0, "t_size": None, "c_size": None, "z_size": None}

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
                        continue
    except Exception:
        traceback.print_exc()
    return summary


def process_tif_file(tif_path: str, out_root: str, skip_existing: bool = True) -> dict:
    tif_path = str(tif_path)
    tif_file = Path(tif_path)
    file_stem = tif_file.stem
    out_dir = compute_out_dir(tif_path, IMAGE_DIR, out_root)
    summary = {"file": tif_path, "out_dir": str(out_dir), "saved": 0, "skipped": 0, "t_size": 1, "c_size": None, "z_size": None}

    try:
        arr = np.asarray(tiff.imread(tif_path))
        if arr.ndim == 3:
            z_size, _, _ = arr.shape
            summary.update({"c_size": 1, "z_size": z_size})
            chan_name = safe_channel_name(DAPI_CHANNEL_INDEX)
            for z_idx in range(z_size):
                out_name = out_dir / f"{file_stem}_t0_z{z_idx}_{chan_name}.tif"
                if skip_existing and out_name.exists():
                    summary["skipped"] += 1
                    continue
                tiff.imwrite(str(out_name), arr[z_idx])
                summary["saved"] += 1
            return summary

        if arr.ndim == 4:
            z_size, c_size, _, _ = arr.shape
            summary.update({"c_size": c_size, "z_size": z_size})
            if DAPI_CHANNEL_INDEX >= c_size:
                raise ValueError(f"DAPI channel index {DAPI_CHANNEL_INDEX} not in array with C={c_size}")
            chan_name = safe_channel_name(DAPI_CHANNEL_INDEX)
            for z_idx in range(z_size):
                out_name = out_dir / f"{file_stem}_t0_z{z_idx}_{chan_name}.tif"
                if skip_existing and out_name.exists():
                    summary["skipped"] += 1
                    continue
                tiff.imwrite(str(out_name), arr[z_idx, DAPI_CHANNEL_INDEX])
                summary["saved"] += 1
            return summary

        raise ValueError(f"Expected 3D (Z,Y,X) or 4D (Z,C,Y,X) TIFF, got shape {arr.shape}")
    except Exception:
        traceback.print_exc()
    return summary


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
                f"[serial] {Path(summary['file']).name}: saved={summary['saved']} "
                f"skipped={summary['skipped']} (C={summary['c_size']}, Z={summary['z_size']})"
            )
        return

    with mp.Pool(processes=PROCESSES, maxtasksperchild=1) as pool:
        for summary in pool.starmap(processor, args):
            print(
                f"[parallel] {Path(summary['file']).name}: saved={summary['saved']} "
                f"skipped={summary['skipped']} (C={summary['c_size']}, Z={summary['z_size']}) "
                f"-> {summary['out_dir']}"
            )


if __name__ == "__main__":
    main()

