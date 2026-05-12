#!/usr/bin/env python3
"""Merge selected columns from a precrop QC CSV into a base QC CSV by ``cell_id``.

Typical use after ``filter_by_intensity.py --from-volume --method otsu_or_bg_voxel:3``
writes ``cell_qc_union_488_560_otsu_or_bg_voxel/qc_features_filtered.csv``:
this script copies only ``otsu_or_bg_voxel_*`` / ``pass_otsu_or_bg_voxel_*``
columns into the rich ``cell_qc_union_488_560/qc_features_filtered.csv`` so
Napari and downstream tools see one wide table without overwriting other
pass/threshold columns.
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
    from config import OUTPUT_DIR, PROJECT_ROOT  # type: ignore[no-redef]


def _merge_column(name: str) -> bool:
    return name.startswith("otsu_or_bg_voxel_") or name.startswith("pass_otsu_or_bg_voxel_")


def _resolve_sample_dir(args: argparse.Namespace) -> Path | None:
    if args.sample_dir is not None:
        return Path(args.sample_dir).resolve()
    if args.data_rel:
        root = (
            Path(OUTPUT_DIR) / args.data_rel
            if OUTPUT_DIR != PROJECT_ROOT / "output"
            else PROJECT_ROOT / "output" / args.data_rel
        )
        return root.resolve()
    return None


def run(base_csv: Path, precrop_csv: Path, output_csv: Path) -> int:
    if not base_csv.is_file():
        print(f"ERROR: base CSV not found: {base_csv}")
        return 1
    if not precrop_csv.is_file():
        print(f"ERROR: precrop CSV not found: {precrop_csv}")
        return 1

    with open(precrop_csv, newline="") as fh:
        precrop_rows = list(csv.DictReader(fh))
    if not precrop_rows:
        print(f"ERROR: no rows in {precrop_csv}")
        return 1

    merge_keys = sorted({k for k in precrop_rows[0].keys() if _merge_column(k)})
    if not merge_keys:
        print(f"ERROR: no mergeable columns (otsu_or_bg_voxel_* / pass_otsu_or_bg_voxel_*) in {precrop_csv}")
        return 1

    precrop_by_id: dict[int, dict[str, str]] = {}
    for row in precrop_rows:
        try:
            cid = int(float(row["cell_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        precrop_by_id[cid] = row

    with open(base_csv, newline="") as fh:
        base_rows = list(csv.DictReader(fh))
    if not base_rows:
        print(f"ERROR: no rows in {base_csv}")
        return 1

    base_fieldnames = list(base_rows[0].keys())
    merged: list[dict[str, str]] = []
    missing = 0
    for row in base_rows:
        out = dict(row)
        try:
            cid = int(float(row["cell_id"]))
        except (KeyError, TypeError, ValueError):
            merged.append(out)
            continue
        pr = precrop_by_id.get(cid)
        if pr is None:
            missing += 1
        for k in merge_keys:
            out[k] = pr.get(k, "") if pr is not None else ""
        merged.append(out)

    fieldnames = list(dict.fromkeys(base_fieldnames + merge_keys))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(output_csv) + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    os.replace(tmp, str(output_csv))
    added = len([k for k in merge_keys if k not in set(base_fieldnames)])
    print(f"Wrote {output_csv}  ({len(merged)} rows, +{added} new columns from precrop)")
    if missing:
        print(f"  WARNING: {missing} base rows had no precrop match (left new cols empty)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-csv", type=Path, default=None, help="Rich qc_features_filtered.csv to extend")
    ap.add_argument("--precrop-csv", type=Path, default=None, help="Precrop filter CSV with otsu_or_bg_voxel_* cols")
    ap.add_argument("--output-csv", type=Path, default=None, help="Default: same as --base-csv")
    ap.add_argument("--sample-dir", type=Path, default=None, help="Sample folder; sets default base/precrop paths")
    ap.add_argument(
        "--data-rel",
        type=str,
        default=None,
        help="Sample folder as OUTPUT_DIR/<data-rel> (alternative to --sample-dir)",
    )
    args = ap.parse_args()

    sd = _resolve_sample_dir(args)
    base = args.base_csv
    precrop = args.precrop_csv
    if sd is not None:
        if base is None:
            base = sd / "cell_qc_union_488_560" / "qc_features_filtered.csv"
        if precrop is None:
            precrop = sd / "cell_qc_union_488_560_otsu_or_bg_voxel" / "qc_features_filtered.csv"
    if base is None or precrop is None:
        ap.error("Provide --base-csv and --precrop-csv, or --sample-dir / --data-rel for defaults")
    out = args.output_csv or base
    return run(base.resolve(), precrop.resolve(), out.resolve())


if __name__ == "__main__":
    sys.exit(main())
