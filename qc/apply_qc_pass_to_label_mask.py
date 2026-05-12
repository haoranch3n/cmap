#!/usr/bin/env python3
"""
Zero labels in a label TIFF where cell_id does not pass ``pass_qc`` in
qc_features_filtered.csv (legacy CSVs may still use ``pass_pixel_intensity``;
that column is accepted as a fallback).

Default input: sample_dir/filtered_642.tif and sample_dir/cell_qc/qc_features_filtered.csv
Default output: sample_dir/filtered_642_pass_otsu_shape.tif (variant-specific)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sys

import numpy as np
import tifffile

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from postprocess import DEFAULT_VARIANT, VALID_VARIANTS, variant_files  # noqa: E402

try:
    from segmentation.config import OUTPUT_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    from config import OUTPUT_DIR, PROJECT_ROOT  # type: ignore[no-redef]


def _resolve_sample_dir(
    sample_dir: Path | None,
    data_rel: str | None,
    output_dir: Path | None,
) -> Path:
    if sample_dir is not None:
        return sample_dir.resolve()
    if output_dir is not None:
        return output_dir.resolve()
    if data_rel:
        root = (
            Path(OUTPUT_DIR) / data_rel
            if OUTPUT_DIR != PROJECT_ROOT / "output"
            else PROJECT_ROOT / "output" / data_rel
        )
        return root.resolve()
    raise ValueError("need --sample-dir, --output-dir, or --data-rel")


def _pass_qc_column(rows: list[dict[str, str]], explicit: str | None = None) -> str:
    if not rows:
        raise ValueError("CSV has no header row")
    keys = rows[0].keys()
    if explicit is not None:
        if explicit not in keys:
            raise ValueError(
                f"CSV missing requested pass column {explicit!r}; "
                f"available columns: {sorted(keys)}"
            )
        return explicit
    # Auto-detect order: prefer the most specific method-named columns first,
    # then fall back to the legacy generic names.
    for candidate in (
        "pass_bg_sigma_488560_shape",
        "pass_bg_sigma_shape",
        "pass_otsu_or_bg_voxel_488560_shape",
        "pass_otsu_or_bg_shape",
        "pass_otsu2_shape",
        "pass_otsu_shape",
        "pass_qc",
        "pass_pixel_intensity",
    ):
        if candidate in keys:
            return candidate
    raise ValueError(
        "CSV missing pass column: expected one of "
        "pass_bg_sigma_488560_shape / pass_bg_sigma_shape / pass_otsu_or_bg_voxel_488560_shape / "
        "pass_otsu_or_bg_shape / pass_otsu2_shape / pass_otsu_shape / pass_qc / pass_pixel_intensity"
    )


def _load_keep_ids(csv_path: Path, pass_column: str | None = None) -> tuple[set[int], str]:
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"CSV has no rows: {csv_path}")
    if "cell_id" not in rows[0]:
        raise ValueError("CSV missing required column: 'cell_id'")
    pass_col = _pass_qc_column(rows, explicit=pass_column)
    keep: set[int] = set()
    for row in rows:
        try:
            if int(float(row[pass_col])) != 1:
                continue
        except (TypeError, ValueError):
            continue
        try:
            keep.add(int(float(row["cell_id"])))
        except (TypeError, ValueError):
            continue
    return keep, pass_col


def apply_mask(mask: np.ndarray, keep_ids: set[int]) -> tuple[np.ndarray, int]:
    """Return filtered mask (same dtype as input) and count of positive labels in volume not in keep_ids."""
    mask = np.asarray(mask)
    if mask.ndim < 2:
        raise ValueError(f"Expected at least 2-D label array, got shape {mask.shape}")
    max_label = int(mask.max())
    if max_label <= 0:
        z = np.zeros_like(mask)
        return z, 0
    lut = np.zeros(max_label + 1, dtype=np.uint32)
    for i in keep_ids:
        if 0 < i <= max_label:
            lut[i] = i
    in_volume = np.unique(mask)
    in_volume = in_volume[in_volume > 0]
    dropped = int(np.sum(~np.isin(in_volume, np.fromiter(keep_ids, dtype=np.int64))))
    idx = mask.astype(np.int64, copy=False)
    out = lut[idx]
    if mask.dtype == np.uint16:
        out = np.clip(out, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    elif mask.dtype == np.uint32:
        out = out.astype(np.uint32)
    else:
        out = out.astype(mask.dtype)
    return out, dropped


def run_sample(
    sample_dir: Path,
    mask_name: str = "filtered_642.tif",
    csv_rel: Path = Path("cell_qc/qc_features_filtered.csv"),
    output_name: str = "filtered_642_pass_otsu_shape.tif",
    overwrite: bool = False,
    dry_run: bool = False,
    pass_column: str | None = None,
) -> int:
    sample_dir = sample_dir.resolve()
    mask_path = sample_dir / mask_name
    csv_path = sample_dir / csv_rel
    out_path = sample_dir / output_name

    if not csv_path.is_file():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1
    if not mask_path.is_file():
        print(f"ERROR: Mask not found: {mask_path}")
        return 1
    if out_path.exists() and not overwrite:
        print(f"SKIP (exists): {out_path}")
        return 0

    keep_ids, pass_col = _load_keep_ids(csv_path, pass_column=pass_column)
    if not keep_ids:
        print(f"No cells with {pass_col} == 1; writing empty mask.")
    mask = tifffile.imread(mask_path)
    out, dropped = apply_mask(np.asarray(mask), keep_ids)
    if dropped:
        print(f"  Warning: {dropped} label value(s) in mask are not in pass set (zeroed).")

    if dry_run:
        print(f"DRY-RUN would write: {out_path} (passing ids: {len(keep_ids)})")
        return 0

    tifffile.imwrite(out_path, out, compression="zlib")
    print(f"Wrote {out_path} (keep {len(keep_ids)} cell ids)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, default=None, help="Sample output directory (contains mask + cell_qc/)")
    ap.add_argument(
        "--data-rel",
        type=str,
        default=None,
        help="Same as other CLIs: sample under output/ (alternative to --sample-dir)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Absolute sample output directory (alternative to --sample-dir)",
    )
    ap.add_argument(
        "--root-dir",
        type=Path,
        action="append",
        default=[],
        help="Batch: parent folder; processes */cell_qc/qc_features_filtered.csv",
    )
    ap.add_argument("--mask-name", type=str, default=None)
    ap.add_argument("--csv-rel", type=Path, default=None)
    ap.add_argument("--output-name", type=str, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--pass-column",
        type=str,
        default=None,
        help=(
            "Explicit name of the pass-flag column in qc_features_filtered.csv "
            "(e.g. pass_bg_sigma_shape, pass_otsu_or_bg_voxel_488560_shape, pass_otsu_or_bg_shape, pass_otsu2_shape, "
            "pass_otsu_shape). Default: auto-detect, preferring the most-specific "
            "method-named column."
        ),
    )
    ap.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default=DEFAULT_VARIANT,
        help=(
            f"Mask variant; sets --mask-name, --output-name, --csv-rel from "
            f"VARIANT_FILES (default: {DEFAULT_VARIANT}). Per-flag overrides win."
        ),
    )
    args = ap.parse_args()

    if args.sample_dir is None and not args.root_dir and args.data_rel is None and args.output_dir is None:
        raise SystemExit("Provide --sample-dir, --output-dir, --data-rel, or --root-dir.")

    v = variant_files(args.variant)
    mask_name = args.mask_name or v["mask"]
    output_name = args.output_name or v["pass_otsu_shape"]
    csv_rel = args.csv_rel or Path(v["cell_qc"]) / "qc_features_filtered.csv"
    print(
        f"Variant: {args.variant}  mask={mask_name}  output={output_name}  csv={csv_rel}"
    )
    args.mask_name = mask_name
    args.output_name = output_name
    args.csv_rel = csv_rel

    if args.sample_dir is not None or args.data_rel or args.output_dir:
        sample_dir = _resolve_sample_dir(args.sample_dir, args.data_rel, args.output_dir)
        return run_sample(
            sample_dir,
            mask_name=args.mask_name,
            csv_rel=args.csv_rel,
            output_name=args.output_name,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            pass_column=args.pass_column,
        )

    failed = 0
    for root in args.root_dir:
        root = root.resolve()
        pattern = f"*/{args.csv_rel.as_posix()}"
        csv_paths = sorted(root.glob(pattern))
        print(f"Root: {root} ({len(csv_paths)} CSV matches)")
        for csv_path in csv_paths:
            sample_dir = csv_path.parents[1]
            mask_path = sample_dir / args.mask_name
            if not mask_path.is_file():
                print(f"SKIP (no mask): {sample_dir}")
                continue
            rc = run_sample(
                sample_dir,
                mask_name=args.mask_name,
                csv_rel=args.csv_rel,
                output_name=args.output_name,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                pass_column=args.pass_column,
            )
            if rc != 0:
                failed += 1
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
