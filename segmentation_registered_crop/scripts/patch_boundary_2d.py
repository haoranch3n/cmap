#!/usr/bin/env python3
"""
Rewrite boundary channel (index 4) in all *_segmented.tif files to use
per-Z-slice 2D boundaries instead of the original 3D volume surface.

Safe to re-run: reads each file, recomputes boundary in-place via .tmp,
then replaces the original atomically.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.segmentation import find_boundaries

OUTPUT_ROOT = "/research/dept/dnb/core_operations/ImageAnalysis/Core/Haoran/cmap/output_registered_crop_seg"


def compute_2d_boundaries(mask_3d: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(mask_3d, dtype=np.uint8)
    for z in range(mask_3d.shape[0]):
        if mask_3d[z].any():
            boundary[z] = find_boundaries(
                mask_3d[z].astype(bool), mode="thick", connectivity=1
            ).astype(np.uint8)
    return boundary


# Patch both per-channel and combined TIFs
patterns = [
    f"{OUTPUT_ROOT}/**/*_segmented.tif",
    f"{OUTPUT_ROOT}/**/*_combined_union.tif",
    f"{OUTPUT_ROOT}/**/*_combined_majority_vote.tif",
]
tifs = sorted({t for p in patterns for t in glob.glob(p, recursive=True)})
print(f"Found {len(tifs)} TIFs to patch.", flush=True)

n_ok = n_err = 0
for i, t in enumerate(tifs, 1):
    try:
        arr = tifffile.imread(t)
        if arr.ndim != 4 or arr.shape[1] != 5:
            print(f"  [SKIP unexpected shape] {Path(t).name}  {arr.shape}", flush=True)
            continue
        mask = (arr[:, 3, :, :] > 0).astype(np.uint8)
        new_boundary = compute_2d_boundaries(mask).astype(np.float32)
        arr[:, 4, :, :] = new_boundary
        tmp = t + ".tmp"
        tifffile.imwrite(tmp, arr, imagej=True, metadata={"axes": "ZCYX"})
        os.replace(tmp, t)
        n_ok += 1
        if n_ok % 100 == 0:
            print(f"  patched {n_ok}/{len(tifs)}...", flush=True)
    except Exception as e:
        import traceback
        print(f"  [ERROR] {t}: {traceback.format_exc()}", flush=True)
        n_err += 1

print(f"\nDone. patched={n_ok} errors={n_err}", flush=True)
sys.exit(1 if n_err else 0)
