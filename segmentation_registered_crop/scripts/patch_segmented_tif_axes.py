#!/usr/bin/env python3
"""
Rewrite all *_segmented.tif files under output_registered_crop_seg/ to add
ImageJ axis metadata (ZCYX), so TIFF viewers correctly show Z-stack + channels
instead of flattening (Z*C) frames as single-channel pages.
"""
import glob, os, sys
import tifffile

OUTPUT_ROOT = "/research/dept/dnb/core_operations/ImageAnalysis/Core/Haoran/cmap/output_registered_crop_seg"

tifs = sorted(glob.glob(f"{OUTPUT_ROOT}/**/*_segmented.tif", recursive=True))
print(f"Found {len(tifs)} segmented TIFs to patch.", flush=True)

n_ok = n_skip = n_err = 0
for t in tifs:
    try:
        arr = tifffile.imread(t)
        if arr.ndim != 4 or arr.shape[1] != 5:
            print(f"  [SKIP unexpected shape] {t}  {arr.shape}", flush=True)
            n_skip += 1
            continue
        tmp = t + ".tmp"
        tifffile.imwrite(tmp, arr, imagej=True, metadata={"axes": "ZCYX"})
        os.replace(tmp, t)
        n_ok += 1
        if n_ok % 50 == 0:
            print(f"  patched {n_ok}/{len(tifs)}...", flush=True)
    except Exception as e:
        print(f"  [ERROR] {t}: {e}", flush=True)
        n_err += 1

print(f"\nDone. patched={n_ok} skipped={n_skip} errors={n_err}")
