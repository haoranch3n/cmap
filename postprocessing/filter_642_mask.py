#!/usr/bin/env python3
"""
Keep only 642nm nuclei that overlap with both 488nm AND 560nm masks
(i.e. nuclei present across all three channels), then write a 4-channel
OME-TIFF:

  channel 0: 642nm original intensity
  channel 1: 488nm original intensity
  channel 2: 560nm original intensity
  channel 3: filtered 642nm mask (only nuclei overlapping all three channels)

A 642 label is kept only if at least one of its voxels overlaps a labelled
voxel in the 488 mask AND at least one overlaps a labelled voxel in the
560 mask.

Usage
-----
    python postprocessing/filter_642_mask.py \
        --data-rel  4_18_25/CGNSample1_Position0_decon_dsr --force

    python postprocessing/filter_642_mask.py \
        --data-dir  /path/to/data/folder \
        --output-dir /path/to/output/folder --force
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

CHANNELS = ["642nm_crop", "488nm_crop", "560nm_crop"]


def _resolve_dirs(args) -> tuple[Path, Path]:
    if args.data_rel:
        data_dir = Path(DATA_DIR) / args.data_rel
        output_dir = (
            Path(OUTPUT_DIR) / args.data_rel
            if OUTPUT_DIR != PROJECT_ROOT / "output"
            else PROJECT_ROOT / "output" / args.data_rel
        )
    elif args.data_dir and args.output_dir:
        data_dir = args.data_dir.resolve()
        output_dir = args.output_dir.resolve()
    else:
        raise SystemExit("Provide either --data-rel or both --data-dir and --output-dir")
    return data_dir, output_dir


def _load_3d(path: Path) -> np.ndarray:
    """Load a TIFF and return a 3-D (Z, Y, X) array."""
    arr = tifffile.imread(str(path))
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D or 4-D, got shape {arr.shape} from {path}")
    return arr


def filter_642_mask(
    mask_642: np.ndarray,
    mask_488: np.ndarray,
    mask_560: np.ndarray,
) -> np.ndarray:
    """Keep only 642 labels that overlap with both 488 and 560 masks."""
    fg_488 = mask_488 > 0
    fg_560 = mask_560 > 0

    labels_overlap_488 = set(np.unique(mask_642[fg_488]))
    labels_overlap_560 = set(np.unique(mask_642[fg_560]))
    labels_overlap_488.discard(0)
    labels_overlap_560.discard(0)

    labels_to_keep = labels_overlap_488 & labels_overlap_560

    all_labels = set(np.unique(mask_642))
    all_labels.discard(0)
    total = len(all_labels)
    print(f"  642 mask: {total} labels total")
    print(f"    overlap with 488: {len(labels_overlap_488)}")
    print(f"    overlap with 560: {len(labels_overlap_560)}")
    print(f"    overlap with both (kept): {len(labels_to_keep)}")
    print(f"    removed: {total - len(labels_to_keep)}")

    filtered = mask_642.copy()
    if labels_to_keep != all_labels:
        remove_mask = ~np.isin(filtered, [0] + list(labels_to_keep))
        filtered[remove_mask] = 0
    return filtered


def run(data_dir: Path, output_dir: Path, force: bool = False) -> int:
    print(f"Data dir:   {data_dir}")
    print(f"Output dir: {output_dir}")

    originals: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}

    for ch in CHANNELS:
        orig_path = data_dir / f"{ch}.tif"
        mask_path = output_dir / ch / "segmentation_3D_masks" / f"{ch}_3D_indexed.tif"
        if not orig_path.exists():
            print(f"ERROR: original not found: {orig_path}")
            return 1
        if not mask_path.exists():
            print(f"ERROR: mask not found: {mask_path}")
            return 1
        print(f"Loading {ch} original: {orig_path}")
        originals[ch] = _load_3d(orig_path).astype(np.float32)
        print(f"Loading {ch} mask:     {mask_path}")
        masks[ch] = _load_3d(mask_path)

    ref_shape = originals[CHANNELS[0]].shape
    for ch in CHANNELS:
        if originals[ch].shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: {CHANNELS[0]} {ref_shape} vs {ch} {originals[ch].shape}"
            )
        if masks[ch].shape != ref_shape:
            z_min = min(ref_shape[0], masks[ch].shape[0])
            print(f"  WARNING: Z mismatch for {ch} mask, truncating to {z_min}")
            masks[ch] = masks[ch][:z_min]
            for k in CHANNELS:
                originals[k] = originals[k][:z_min]
                if masks[k].shape[0] > z_min:
                    masks[k] = masks[k][:z_min]
            ref_shape = originals[CHANNELS[0]].shape

    print("\nFiltering 642 mask...")
    filtered_mask = filter_642_mask(masks["642nm_crop"], masks["488nm_crop"], masks["560nm_crop"])
    remaining = len(np.unique(filtered_mask)) - 1  # exclude 0
    print(f"  Remaining labels in filtered 642 mask: {remaining}\n")

    combined = np.stack(
        [
            originals["642nm_crop"],
            originals["488nm_crop"],
            originals["560nm_crop"],
            filtered_mask.astype(np.float32),
        ],
        axis=1,
    )  # (Z, 4, Y, X)

    out_path = output_dir / "filtered_642_combined.tif"
    if out_path.exists() and not force:
        print(f"SKIP: {out_path} already exists (use --force to overwrite)")
        return 0

    nbytes = combined.nbytes
    print(f"Output shape: {combined.shape}  dtype={combined.dtype}  size={nbytes / 1e9:.2f} GB")
    tmp_path = str(out_path) + ".tmp"
    print(f"Writing OME-TIFF / BigTIFF: {out_path}")
    tifffile.imwrite(
        tmp_path,
        combined,
        bigtiff=True,
        ome=True,
        photometric="minisblack",
        compression="zlib",
        metadata={
            "axes": "ZCYX",
            "Channel": {
                "Name": [
                    "642_Original",
                    "488_Original",
                    "560_Original",
                    "642_Mask_Filtered",
                ]
            },
        },
    )
    import os
    os.replace(tmp_path, str(out_path))
    print("Done.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-rel", type=str, default=None,
        help="Relative path under data/ and output/ (e.g. 4_18_25/CGNSample1...)",
    )
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = ap.parse_args()

    data_dir, output_dir = _resolve_dirs(args)
    return run(data_dir, output_dir, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
