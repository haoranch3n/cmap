#!/usr/bin/env python3
"""
Combine original volume and indexed segmentation mask into a two-channel TIF.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tifffile

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from segmentation.config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    from config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT

_IMAGEJ_MAX_BYTES = 3_900_000_000


def _find_original_volume(data_dir: Path, stem: str) -> Path:
    prefix = stem.replace("_crop", "")
    reg_matches = sorted(data_dir.glob(f"reg_{prefix}*.tif"))
    if len(reg_matches) == 1:
        return reg_matches[0]
    if len(reg_matches) > 1:
        warped = [m for m in reg_matches if "Warped" in m.name]
        if len(warped) == 1:
            return warped[0]
        raise ValueError(f"Multiple reg_{prefix}*.tif in {data_dir}: {reg_matches}")
    direct = data_dir / f"{stem}.tif"
    if direct.exists():
        return direct
    raise FileNotFoundError(f"No original for {stem} in {data_dir}")


def find_pairs(data_dir: Path, output_dir: Path) -> list[dict]:
    pairs = []
    if not output_dir.is_dir():
        return pairs
    for stem_dir in sorted(output_dir.iterdir()):
        if not stem_dir.is_dir():
            continue
        stem = stem_dir.name
        if stem.startswith("_"):
            continue
        try:
            original = _find_original_volume(data_dir, stem)
        except (FileNotFoundError, ValueError) as exc:
            print(f"SKIP {stem}: {exc}")
            continue
        mask = stem_dir / "segmentation_3D_masks" / f"{stem}_3D_indexed.tif"
        if not mask.exists():
            print(f"SKIP {stem}: indexed mask not found at {mask}")
            continue
        out = stem_dir / f"{stem}_combined.tif"
        pairs.append(dict(stem=stem, original=original, mask=mask, out=out))
    return pairs


def _load_3d(path: Path, name: str) -> np.ndarray:
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
        raise ValueError(f"XY mismatch: original ({y_orig},{x_orig}) vs mask ({y_mask},{x_mask})")
    if z_orig != z_mask:
        z_min = min(z_orig, z_mask)
        print(f"  WARNING: Z mismatch ({z_orig} vs {z_mask}), truncating to {z_min}")
        orig = orig[:z_min]
        mask = mask[:z_min]

    combined = np.stack([orig, mask.astype(np.float32)], axis=1)
    nbytes = combined.nbytes
    tmp_path = str(out_path) + ".tmp"
    if nbytes < _IMAGEJ_MAX_BYTES:
        tifffile.imwrite(
            tmp_path,
            combined,
            imagej=True,
            photometric="minisblack",
            compression="zlib",
            metadata={"axes": "ZCYX", "mode": "grayscale"},
        )
    else:
        tifffile.imwrite(
            tmp_path,
            combined,
            bigtiff=True,
            ome=True,
            photometric="minisblack",
            compression="zlib",
            metadata={"axes": "ZCYX", "Channel": {"Name": ["Original", "Mask"]}},
        )
    os.replace(tmp_path, str(out_path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-rel", type=str, default=None, help="Relative path under data/ and output/")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="Overwrite existing combined files")
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

    pairs = find_pairs(data_dir, output_dir)
    if not pairs:
        print("No valid original + mask pairs found.")
        return 1

    for p in pairs:
        old_ome = p["out"].with_name(p["out"].stem + ".ome.tif")
        if old_ome.exists() and old_ome != p["out"]:
            old_ome.unlink()
        if p["out"].exists():
            if args.force:
                p["out"].unlink()
            else:
                print(f"  SKIP: already exists {p['out']}  (use --force to overwrite)")
                continue
        combine(p["original"], p["mask"], p["out"])

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

