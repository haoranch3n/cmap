#!/usr/bin/env python3
"""
Keep only 642nm nuclei that overlap with both 488nm AND 560nm masks
(i.e. nuclei present across all three channels).

**Writes** (by default):

- ``<output>/642nm_crop/segmentation_3D_masks/642nm_crop_3D_indexed_filtered.tif`` —
  filtered 3D label volume (``uint16``).
- ``<output>/filtered_642.tif`` — same volume as above (short name at merge root).
- ``<output>/filtered_642_combined.tif`` — 4-channel OME BigTIFF (Z, C, Y, X):
  642 / 488 / 560 originals + filtered mask. Use ``--skip-combined`` to skip (large file).

With ``--export-all-anchors``, also write the same triple-overlap rule with 560nm or
488nm as the anchor (``filtered_560.tif``, ``filtered_488.tif``, and matching
``*_3D_indexed_filtered.tif`` under each crop tree). Downstream steps still use 642
by default.

Passthrough from ``pipelines/run_analysis_pipeline.py``::

    --passthrough --data-rel <rel> --export-all-anchors

A 642 label is kept only if at least one of its voxels overlaps a labelled
voxel in the 488 mask AND at least one overlaps a labelled voxel in the
560 mask.
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

CHANNELS = ["642nm_crop", "488nm_crop", "560nm_crop"]

# channel folder name -> short stem for filtered_<stem>.tif at merge root
_CHANNEL_TO_SHORT_TIFF = {"642nm_crop": "642", "488nm_crop": "488", "560nm_crop": "560"}


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
    arr = tifffile.imread(str(path))
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D or 4-D, got shape {arr.shape} from {path}")
    return arr


def _find_indexed_mask(output_dir: Path, ch: str) -> Path:
    seg_dir = output_dir / ch / "segmentation_3D_masks"
    if not seg_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {seg_dir}")
    direct = seg_dir / f"{ch}_3D_indexed.tif"
    if direct.is_file():
        return direct
    matches = sorted(p for p in seg_dir.rglob("*_3D_indexed.tif") if p.is_file())
    if not matches:
        raise FileNotFoundError(f"No *_3D_indexed.tif under {seg_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple indexed masks under {seg_dir}; expected one: {matches}")
    return matches[0]


def filter_anchor_by_other_two(
    anchor: np.ndarray,
    other_a: np.ndarray,
    other_b: np.ndarray,
    *,
    anchor_label: str,
    other_a_label: str,
    other_b_label: str,
) -> np.ndarray:
    """Keep anchor labels that overlap foreground in both other_a and other_b."""
    fg_a = other_a > 0
    fg_b = other_b > 0
    labels_overlap_a = set(np.unique(anchor[fg_a]))
    labels_overlap_b = set(np.unique(anchor[fg_b]))
    labels_overlap_a.discard(0)
    labels_overlap_b.discard(0)
    labels_to_keep = labels_overlap_a & labels_overlap_b

    all_labels = set(np.unique(anchor))
    all_labels.discard(0)
    total = len(all_labels)
    print(f"  {anchor_label} mask: {total} labels total")
    print(f"    overlap with {other_a_label}: {len(labels_overlap_a)}")
    print(f"    overlap with {other_b_label}: {len(labels_overlap_b)}")
    print(f"    overlap with both (kept): {len(labels_to_keep)}")
    print(f"    removed: {total - len(labels_to_keep)}")

    filtered = anchor.copy()
    if labels_to_keep != all_labels:
        remove_mask = ~np.isin(filtered, [0] + list(labels_to_keep))
        filtered[remove_mask] = 0
    return filtered


def filter_642_mask(mask_642: np.ndarray, mask_488: np.ndarray, mask_560: np.ndarray) -> np.ndarray:
    return filter_anchor_by_other_two(
        mask_642,
        mask_488,
        mask_560,
        anchor_label="642",
        other_a_label="488",
        other_b_label="560",
    )


def _crop_zyx_to_ref(arr: np.ndarray, ref: tuple[int, int, int], label: str) -> np.ndarray:
    rz, ry, rx = ref
    if arr.shape[1:] != (ry, rx):
        raise ValueError(f"{label} spatial shape {arr.shape[1:]} != ref {(ry, rx)}")
    if arr.shape[0] < rz:
        raise ValueError(f"{label} Z={arr.shape[0]} < ref Z={rz}")
    return arr[:rz]


def _write_filtered_volume_pair(
    output_dir: Path,
    ch_folder: str,
    filtered: np.ndarray,
    force: bool,
) -> None:
    """Write ``<ch>_3D_indexed_filtered.tif`` and ``filtered_<short>.tif``."""
    short = _CHANNEL_TO_SHORT_TIFF[ch_folder]
    filtered_out = output_dir / ch_folder / "segmentation_3D_masks" / f"{ch_folder}_3D_indexed_filtered.tif"
    filtered_short = output_dir / f"filtered_{short}.tif"

    if not filtered_out.exists() or force:
        filtered_out.parent.mkdir(parents=True, exist_ok=True)
        tmp_filtered = str(filtered_out) + ".tmp"
        tifffile.imwrite(
            tmp_filtered,
            filtered.astype(np.uint16),
            photometric="minisblack",
            compression="zlib",
            metadata={"axes": "ZYX"},
        )
        os.replace(tmp_filtered, str(filtered_out))
    else:
        print(f"SKIP (exists): {filtered_out}")

    if not filtered_short.exists() or force:
        tmp_short = str(filtered_short) + ".tmp"
        tifffile.imwrite(
            tmp_short,
            filtered.astype(np.uint16),
            photometric="minisblack",
            compression="zlib",
            metadata={"axes": "ZYX"},
        )
        os.replace(tmp_short, str(filtered_short))
    else:
        print(f"SKIP (exists): {filtered_short}")


def run(
    data_dir: Path,
    output_dir: Path,
    force: bool = False,
    skip_combined: bool = False,
    export_all_anchors: bool = False,
) -> int:
    print(f"Data dir:   {data_dir}")
    print(f"Output dir: {output_dir}")
    masks: dict[str, np.ndarray] = {}
    for ch in CHANNELS:
        try:
            _find_original_volume(data_dir, ch)
            mask_path = _find_indexed_mask(output_dir, ch)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1
        print(f"Loading {ch} mask: {mask_path}")
        masks[ch] = _load_3d(mask_path)

    ref_shape = masks[CHANNELS[0]].shape
    for ch in CHANNELS:
        if masks[ch].shape != ref_shape:
            if masks[ch].shape[1:] != ref_shape[1:]:
                raise ValueError(f"Mask XY mismatch: {CHANNELS[0]} {ref_shape} vs {ch} {masks[ch].shape}")
            z_min = min(ref_shape[0], masks[ch].shape[0])
            print(f"  WARNING: Z mismatch for {ch} mask, truncating all masks to {z_min}")
            for k in CHANNELS:
                masks[k] = masks[k][:z_min]
            ref_shape = masks[CHANNELS[0]].shape

    print("\nFiltering 642 mask...")
    filtered_mask = filter_642_mask(masks["642nm_crop"], masks["488nm_crop"], masks["560nm_crop"])
    remaining = len(np.unique(filtered_mask)) - 1
    print(f"  Remaining labels in filtered 642 mask: {remaining}\n")

    ref_shape = filtered_mask.shape
    _write_filtered_volume_pair(output_dir, "642nm_crop", filtered_mask, force)

    if export_all_anchors:
        print("\nFiltering 560 mask (overlap with 488 and 642)...")
        f560 = filter_anchor_by_other_two(
            masks["560nm_crop"],
            masks["488nm_crop"],
            masks["642nm_crop"],
            anchor_label="560",
            other_a_label="488",
            other_b_label="642",
        )
        print(f"  Remaining labels in filtered 560 mask: {len(np.unique(f560)) - 1}\n")
        _write_filtered_volume_pair(output_dir, "560nm_crop", f560, force)

        print("\nFiltering 488 mask (overlap with 560 and 642)...")
        f488 = filter_anchor_by_other_two(
            masks["488nm_crop"],
            masks["560nm_crop"],
            masks["642nm_crop"],
            anchor_label="488",
            other_a_label="560",
            other_b_label="642",
        )
        print(f"  Remaining labels in filtered 488 mask: {len(np.unique(f488)) - 1}\n")
        _write_filtered_volume_pair(output_dir, "488nm_crop", f488, force)

    combined_out = output_dir / "filtered_642_combined.tif"
    if skip_combined:
        print("SKIP: --skip-combined set; not writing filtered_642_combined.tif")
        print("Done.")
        return 0
    if combined_out.exists() and not force:
        print(f"SKIP (exists): {combined_out}")
        print("Done.")
        return 0

    originals: dict[str, np.ndarray] = {}
    for ch in CHANNELS:
        orig_path = _find_original_volume(data_dir, ch)
        o = _load_3d(orig_path).astype(np.float32)
        originals[ch] = _crop_zyx_to_ref(o, ref_shape, ch)

    combined = np.stack(
        [
            originals["642nm_crop"],
            originals["488nm_crop"],
            originals["560nm_crop"],
            filtered_mask.astype(np.float32),
        ],
        axis=1,
    )
    tmp_path = str(combined_out) + ".tmp"
    tifffile.imwrite(
        tmp_path,
        combined,
        bigtiff=True,
        ome=True,
        photometric="minisblack",
        compression="zlib",
        metadata={
            "axes": "ZCYX",
            "Channel": {"Name": ["642_Original", "488_Original", "560_Original", "642_Mask_Filtered"]},
        },
    )
    os.replace(tmp_path, str(combined_out))
    print("Done.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-rel", type=str, default=None, help="Relative path under data/ and output/")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="Overwrite existing output")
    ap.add_argument("--skip-combined", action="store_true", help="Do not write filtered_642_combined.tif")
    ap.add_argument(
        "--export-all-anchors",
        action="store_true",
        help="Also write filtered_560.tif / filtered_488.tif (triple overlap, same rule as 642)",
    )
    args = ap.parse_args()
    data_dir, output_dir = _resolve_dirs(args)
    return run(
        data_dir,
        output_dir,
        force=args.force,
        skip_combined=args.skip_combined,
        export_all_anchors=args.export_all_anchors,
    )


if __name__ == "__main__":
    sys.exit(main())
