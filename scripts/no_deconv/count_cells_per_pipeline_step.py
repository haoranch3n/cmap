#!/usr/bin/env python3
"""
Scan per-sample CMAP artifacts and write one CSV row per sample with approximate
cell / object counts at each materialized pipeline stage (filtered_642 variant).

Stages (columns) — all counts are integers or empty if the file is missing:

- Segmentation (indexed masks, unique positive label ids):
  n_seg_488, n_seg_560, n_seg_642
- After ``postprocess/filter_642_mask.py`` (triple overlap, before intensity/shape QC):
  n_filtered_642
- After ``qc/apply_qc_pass_to_label_mask.py`` (labels surviving QC in the pass mask):
  n_pass_otsu_shape
- After ``features/crop_cells.py`` (rows in ``cell_boxing_filtered/summary.csv``):
  n_cropped_summary
- After ``features/extract_features.py``:
  n_qc_features
- After ``qc/merge_qc_features_filtered.py`` (final merged table; overwrites the
  precrop-only ``qc_features_filtered.csv`` from ``filter_by_intensity``):
  n_qc_features_filtered_merged

The precrop-only row count from ``filter_by_intensity.py`` is not recoverable
from disk after merge unless you archive that CSV separately.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from postprocess import VALID_VARIANTS, variant_files  # noqa: E402


def _unique_positive_labels(path: Path) -> int | None:
    """Count distinct positive label ids (I/O-friendly: Z-slice TIFFs page-by-page)."""
    if not path.is_file():
        return None
    with tifffile.TiffFile(str(path)) as tf:
        if tf.series and tf.pages:
            s0 = tf.series[0]
            if len(s0.shape) == 3 and len(tf.pages) == int(s0.shape[0]):
                h, w = int(s0.shape[1]), int(s0.shape[2])
                if tf.pages[0].shape == (h, w):
                    bc = np.zeros(1, dtype=np.int64)
                    for page in tf.pages:
                        plane = page.asarray()
                        mv = int(plane.max())
                        if mv <= 0:
                            continue
                        if mv + 1 > bc.size:
                            bc = np.resize(bc, mv + 1)
                        bc += np.bincount(plane.ravel(), minlength=bc.size)
                    if bc.size <= 1:
                        return 0
                    return int(np.count_nonzero(bc[1:]))
        arr = tf.asarray(out="memmap")
    m = int(np.max(arr))
    if m <= 0:
        return 0
    bc = np.bincount(arr.ravel(), minlength=m + 1)
    return int(np.count_nonzero(bc[1:]))


def _csv_nrows(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open(newline="") as fh:
        r = csv.reader(fh)
        try:
            next(r)
        except StopIteration:
            return 0
        return sum(1 for _ in r)


def _scan_sample(sample_dir: Path, variant: str) -> dict[str, str | int | None]:
    v = variant_files(variant)
    rel_root = sample_dir.name
    batch = sample_dir.parent.name

    seg488 = sample_dir / "488nm_crop/segmentation_3D_masks/488nm_crop_3D_indexed.tif"
    seg560 = sample_dir / "560nm_crop/segmentation_3D_masks/560nm_crop_3D_indexed.tif"
    seg642 = sample_dir / "642nm_crop/segmentation_3D_masks/642nm_crop_3D_indexed.tif"

    f642 = sample_dir / v["mask"]
    pass_mask = sample_dir / v["pass_otsu_shape"]
    summary = sample_dir / v["cell_box_filtered"] / "summary.csv"
    qc_dir = sample_dir / v["cell_qc"]
    qfeat = qc_dir / "qc_features.csv"
    qfilt = qc_dir / "qc_features_filtered.csv"

    return {
        "batch": batch,
        "sample": rel_root,
        "sample_dir": str(sample_dir),
        "variant": variant,
        "n_seg_488": _unique_positive_labels(seg488),
        "n_seg_560": _unique_positive_labels(seg560),
        "n_seg_642": _unique_positive_labels(seg642),
        "n_filtered_642": _unique_positive_labels(f642),
        "n_pass_otsu_shape": _unique_positive_labels(pass_mask),
        "n_cropped_summary": _csv_nrows(summary),
        "n_qc_features": _csv_nrows(qfeat),
        "n_qc_features_filtered_merged": _csv_nrows(qfilt),
    }


def _fmt(v: str | int | None) -> str:
    if v is None:
        return ""
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="e.g. .../cmap/output_no_deconv (expects <batch>/<sample>/ per sample)",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Path to write the summary CSV (parent dirs created).",
    )
    ap.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default="filtered_642",
        help="Mask variant for downstream paths (default: filtered_642).",
    )
    args = ap.parse_args()

    root: Path = args.output_root.resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1

    samples = sorted(d for d in root.glob("*/*") if d.is_dir())
    if not samples:
        print(f"ERROR: no sample dirs under {root}", file=sys.stderr)
        return 1

    fieldnames = [
        "batch",
        "sample",
        "sample_dir",
        "variant",
        "n_seg_488",
        "n_seg_560",
        "n_seg_642",
        "n_filtered_642",
        "n_pass_otsu_shape",
        "n_cropped_summary",
        "n_qc_features",
        "n_qc_features_filtered_merged",
    ]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_csv.with_suffix(args.out_csv.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        fh.flush()
        n = len(samples)
        for i, sd in enumerate(samples, start=1):
            print(f"[{i}/{n}] {sd.parent.name}/{sd.name}", flush=True)
            row = _scan_sample(sd, args.variant)
            w.writerow({k: _fmt(row[k]) for k in fieldnames})
            fh.flush()

    tmp.replace(args.out_csv)
    print(f"Wrote {args.out_csv}  ({len(samples)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
