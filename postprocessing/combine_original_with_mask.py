#!/usr/bin/env python3
"""
Combine the original volume with its 3D indexed segmentation mask into a
two-channel ImageJ hyperstack TIF (grayscale).

Output shape: (Z, 2, Y, X), float32
  - channel 0: original intensity volume
  - channel 1: 3D indexed segmentation mask (label IDs)

Both channels are displayed in grayscale mode.

Usage
-----
    python postprocessing/combine_original_with_mask.py \
        --data-rel  4_18_25/CGNSample1_Position0_decon_dsr --force

Alternatively specify ``--data-dir`` and ``--output-dir`` directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline_config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT

_IMAGEJ_MAX_BYTES = 3_900_000_000  # conservative 4 GB limit


def find_pairs(data_dir: Path, output_dir: Path) -> list[dict]:
    """Return list of {stem, original, mask, out} dicts."""
    pairs = []
    if not output_dir.is_dir():
        return pairs
    for stem_dir in sorted(output_dir.iterdir()):
        if not stem_dir.is_dir():
            continue
        stem = stem_dir.name
        if stem.startswith("_"):
            continue
        original = data_dir / f"{stem}.tif"
        mask = stem_dir / "segmentation_3D_masks" / f"{stem}_3D_indexed.tif"
        if not original.exists():
            print(f"SKIP {stem}: original not found at {original}")
            continue
        if not mask.exists():
            print(f"SKIP {stem}: indexed mask not found at {mask}")
            continue
        out = stem_dir / f"{stem}_combined.tif"
        pairs.append(dict(stem=stem, original=original, mask=mask, out=out))
    return pairs


def _load_3d(path: Path, name: str) -> np.ndarray:
    """Load a TIFF and return a 3-D (Z, Y, X) array."""
    print(f"  Loading {name}: {path}")
    arr = tifffile.imread(str(path))
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D or 4-D {name}, got shape {arr.shape}")
    return arr


def combine(original_path: Path, mask_path: Path, out_path: Path) -> None:
    orig = _load_3d(original_path, "original").astype(np.float32)
    mask = _load_3d(mask_path, "mask")

    z_orig, y_orig, x_orig = orig.shape
    z_mask, y_mask, x_mask = mask.shape

    if (y_orig, x_orig) != (y_mask, x_mask):
        raise ValueError(
            f"XY mismatch: original ({y_orig},{x_orig}) vs mask ({y_mask},{x_mask})"
        )
    if z_orig != z_mask:
        z_min = min(z_orig, z_mask)
        print(f"  WARNING: Z mismatch ({z_orig} vs {z_mask}), truncating to {z_min}")
        orig = orig[:z_min]
        mask = mask[:z_min]

    mask_f32 = mask.astype(np.float32)
    combined = np.stack([orig, mask_f32], axis=1)  # (Z, 2, Y, X)

    print(f"  orig  range : [{orig.min():.2f}, {orig.max():.2f}]")
    print(f"  mask  labels: {int(mask.max())} cells")
    print(f"  output shape: {combined.shape}  dtype={combined.dtype}")

    nbytes = combined.nbytes
    tmp_path = str(out_path) + ".tmp"
    if nbytes < _IMAGEJ_MAX_BYTES:
        print(f"  Writing ImageJ hyperstack ({nbytes / 1e9:.2f} GB): {out_path}")
        tifffile.imwrite(
            tmp_path,
            combined,
            imagej=True,
            photometric="minisblack",
            compression="zlib",
            metadata={"axes": "ZCYX", "mode": "grayscale"},
        )
    else:
        print(f"  Writing OME-TIFF / BigTIFF ({nbytes / 1e9:.2f} GB): {out_path}")
        tifffile.imwrite(
            tmp_path,
            combined,
            bigtiff=True,
            ome=True,
            photometric="minisblack",
            compression="zlib",
            metadata={
                "axes": "ZCYX",
                "Channel": {"Name": ["Original", "Mask"]},
            },
        )
    import os
    os.replace(tmp_path, str(out_path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-rel", type=str, default=None,
                    help="Relative path under data/ and output/ (e.g. 4_18_25/CGNSample1...)")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Explicit path to the folder containing original .tif files")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Explicit path to the per-image output root")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing combined files")
    args = ap.parse_args()

    if args.data_rel:
        data_dir = Path(DATA_DIR) / args.data_rel
        output_dir = Path(OUTPUT_DIR) / args.data_rel if OUTPUT_DIR != PROJECT_ROOT / "output" else PROJECT_ROOT / "output" / args.data_rel
    elif args.data_dir and args.output_dir:
        data_dir = args.data_dir.resolve()
        output_dir = args.output_dir.resolve()
    else:
        ap.error("Provide either --data-rel or both --data-dir and --output-dir")
        return 1

    print(f"Data dir:   {data_dir}")
    print(f"Output dir: {output_dir}")

    pairs = find_pairs(data_dir, output_dir)
    if not pairs:
        print("No valid original + mask pairs found.")
        return 1

    print(f"Found {len(pairs)} pair(s): {[p['stem'] for p in pairs]}\n")
    for p in pairs:
        print(f"[{p['stem']}]")
        # clean up old .ome.tif left from previous naming convention
        old_ome = p["out"].with_name(p["out"].stem + ".ome.tif")
        if old_ome.exists() and old_ome != p["out"]:
            old_ome.unlink()
            print(f"  Removed old file: {old_ome}")
        if p["out"].exists():
            if args.force:
                print(f"  Overwriting: {p['out']}")
                p["out"].unlink()
            else:
                print(f"  SKIP: already exists {p['out']}  (use --force to overwrite)")
                continue
        combine(p["original"], p["mask"], p["out"])
        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
