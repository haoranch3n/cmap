"""
Stack per-Z 2D segmentation masks into a single 3D (Z, Y, X) volume TIFF.

Adapted from segmentation/assemble3d/stack_2d_planes.py for use with
per-crop per-channel directory layouts.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import tifffile

_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))


def tz_sort_key(name: str) -> tuple:
    m = re.search(r"_t(\d+)_z(\d+)$", name)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"_z(\d+)$", name)
    return (0, int(m.group(1)) if m else -1)


def should_skip(out_path: str) -> bool:
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def stack_volume(seg2d_dir: Path, stacked_dir: Path, out_dtype=np.uint16) -> str | None:
    """
    Walk ``seg2d_dir`` for per-Z subdirs that each contain a ``*_final_mask.tif``,
    sort them by (t, z), stack into a single (Z, Y, X) TIFF, and write to
    ``stacked_dir/<stem>_2D_stacked.tif``.

    Returns the output path on success, or None if no masks were found.
    """
    seg2d_dir = Path(seg2d_dir)
    stacked_dir = Path(stacked_dir)
    stacked_dir.mkdir(parents=True, exist_ok=True)

    # Collect all z-folders containing a _final_mask.tif
    z_folders: list[str] = []
    for entry in os.scandir(seg2d_dir):
        if not entry.is_dir():
            continue
        mask_path = os.path.join(entry.path, f"{entry.name}_final_mask.tif")
        if os.path.exists(mask_path):
            z_folders.append(entry.name)

    if not z_folders:
        return None

    z_folders_sorted = sorted(z_folders, key=tz_sort_key)
    stem = seg2d_dir.name
    out_path = str(stacked_dir / f"{stem}_2D_stacked.tif")

    if should_skip(out_path):
        return out_path

    volume = []
    missing = 0
    for zf in z_folders_sorted:
        tif_path = os.path.join(seg2d_dir, zf, f"{zf}_final_mask.tif")
        if os.path.exists(tif_path):
            volume.append(tifffile.imread(tif_path))
        else:
            missing += 1

    if not volume:
        return None

    vol = np.stack(volume, axis=0)
    tifffile.imwrite(out_path, vol.astype(out_dtype))
    if missing:
        print(f"  [warn] {missing} missing slice(s) for {stem}")
    return out_path
