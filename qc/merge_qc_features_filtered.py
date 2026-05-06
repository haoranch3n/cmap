#!/usr/bin/env python3
"""
Join precrop intensity pass columns into post-crop qc_features.csv.

After ``filter_by_intensity.py --from-volume``, ``cell_qc/.../qc_features_filtered.csv``
holds minimal rows (cell_id, mean_*, threshold_*, pass_*, px_*, pass_intensity,
pass_qc). After ``extract_features.py`` on filtered crops,
``qc_features.csv`` holds the full per-cell metrics. This script writes a new
``qc_features_filtered.csv`` with the full metrics plus the pass/threshold
columns from the precrop file (matched by cell_id).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from segmentation.config import OUTPUT_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    from config import OUTPUT_DIR, PROJECT_ROOT

from postprocess import DEFAULT_VARIANT, VALID_VARIANTS, variant_files

CHANNEL_NAMES = ["642", "488", "560"]

PASS_SUFFIX_KEYS: list[str] = []
for ch in CHANNEL_NAMES:
    PASS_SUFFIX_KEYS.extend(
        [f"threshold_{ch}", f"pass_{ch}", f"px_threshold_{ch}", f"px_pass_{ch}"]
    )
PASS_SUFFIX_KEYS.extend(
    [
        "shape_volume",
        "shape_bbox_aspect",
        "shape_fill_ratio",
        "shape_inertia_ratio",
        "shape_erode2_n_cc",
        "pass_shape",
        "pass_intensity",
        "pass_qc",
    ]
)


def _resolve_dirs(args) -> Path:
    if args.data_rel:
        return (
            Path(OUTPUT_DIR) / args.data_rel
            if OUTPUT_DIR != PROJECT_ROOT / "output"
            else PROJECT_ROOT / "output" / args.data_rel
        ).resolve()
    if args.output_dir:
        return args.output_dir.resolve()
    raise SystemExit("Provide --data-rel or --output-dir")


def run(output_dir: Path, variant: str) -> int:
    v = variant_files(variant)
    qc_dir = output_dir / v["cell_qc"]
    full_path = qc_dir / "qc_features.csv"
    precrop_path = qc_dir / "qc_features_filtered.csv"
    out_path = precrop_path

    if not full_path.is_file():
        print(f"ERROR: {full_path} not found (run extract_features.py first)")
        return 1
    if not precrop_path.is_file():
        print(f"ERROR: {precrop_path} not found (run filter_by_intensity --from-volume first)")
        return 1

    with open(precrop_path, newline="") as fh:
        precrop_rows = list(csv.DictReader(fh))
    if not precrop_rows:
        print(f"ERROR: no rows in {precrop_path}")
        return 1

    with open(full_path, newline="") as fh:
        full_rows = list(csv.DictReader(fh))
    if not full_rows:
        print(f"ERROR: no rows in {full_path}")
        return 1

    precrop_by_id: dict[int, dict[str, str]] = {}
    for row in precrop_rows:
        try:
            cid = int(float(row["cell_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        precrop_by_id[cid] = row

    merged: list[dict[str, str]] = []
    missing_precrop = 0
    for row in full_rows:
        try:
            cid = int(float(row["cell_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        pr = precrop_by_id.get(cid)
        if pr is None:
            missing_precrop += 1
            continue
        out_row = dict(row)
        for k in PASS_SUFFIX_KEYS:
            if k in pr:
                out_row[k] = pr[k]
        if "pass_qc" not in out_row and "pass_pixel_intensity" in pr:
            out_row["pass_qc"] = pr["pass_pixel_intensity"]
        merged.append(out_row)

    if not merged:
        print("ERROR: no rows could be merged (cell_id mismatch?)")
        return 1
    if missing_precrop:
        print(f"  WARNING: {missing_precrop} full rows had no precrop match (skipped)")

    base_keys = list(merged[0].keys())
    extra = [k for k in PASS_SUFFIX_KEYS if k not in base_keys]
    fieldnames = base_keys + extra

    qc_dir.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    os.replace(tmp, str(out_path))
    print(f"Wrote {out_path}  ({len(merged)} rows, {len(fieldnames)} columns)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-rel", type=str, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default=DEFAULT_VARIANT,
        help=f"Mask variant (default: {DEFAULT_VARIANT})",
    )
    args = ap.parse_args()
    output_dir = _resolve_dirs(args)
    return run(output_dir, variant=args.variant)


if __name__ == "__main__":
    sys.exit(main())
