#!/usr/bin/env python3
"""
Filter cells by per-channel mean intensity and append pass/fail to QC CSV.
"""
from __future__ import annotations

import argparse
import csv
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

CHANNEL_NAMES = ["642", "488", "560"]
CHANNEL_INDICES = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}


def _resolve_dirs(args) -> tuple[Path, Path]:
    if args.data_rel:
        data_dir = Path(DATA_DIR) / args.data_rel
        output_dir = (
            Path(OUTPUT_DIR) / args.data_rel
            if OUTPUT_DIR != PROJECT_ROOT / "output"
            else PROJECT_ROOT / "output" / args.data_rel
        )
    elif args.output_dir:
        data_dir = args.output_dir.resolve()
        output_dir = args.output_dir.resolve()
    else:
        raise SystemExit("Provide either --data-rel or --output-dir")
    return data_dir, output_dir


def _parse_method(method_str: str) -> tuple[str, float | None]:
    if method_str == "log_otsu":
        return "log_otsu", None
    if method_str == "otsu":
        return "otsu", None
    if method_str.startswith("percentile:"):
        val = float(method_str.split(":", 1)[1])
        if not 0 < val < 100:
            raise ValueError(f"Percentile must be in (0, 100), got {val}")
        return "percentile", val
    if method_str.startswith("fixed:"):
        return "fixed", float(method_str.split(":", 1)[1])
    raise ValueError("Unknown method. Use 'log_otsu', 'otsu', 'percentile:N', or 'fixed:V'.")


def _compute_threshold(values: np.ndarray, method: str, param: float | None) -> float:
    if len(values) < 2:
        return 0.0
    from skimage.filters import threshold_otsu

    if method == "log_otsu":
        pos = values[values > 0]
        if len(pos) < 2:
            return 0.0
        try:
            return float(np.exp(float(threshold_otsu(np.log(pos)))))
        except ValueError:
            return 0.0
    if method == "otsu":
        try:
            return float(threshold_otsu(values))
        except ValueError:
            return 0.0
    if method == "percentile":
        return float(np.percentile(values, param))
    if method == "fixed":
        return param
    raise ValueError(f"Unknown method: {method}")


def _compute_pixel_thresholds(box_dir: Path) -> dict[str, float]:
    from skimage.filters import threshold_otsu

    cell_tifs = sorted(box_dir.glob("cell_*.tif"))
    if not cell_tifs:
        return {ch: 0.0 for ch in CHANNEL_NAMES}
    pixel_pools: dict[str, list[np.ndarray]] = {ch: [] for ch in CHANNEL_NAMES}
    for tif_path in cell_tifs:
        crop = tifffile.imread(str(tif_path))
        if crop.ndim != 4 or crop.shape[1] < 4:
            continue
        mask = crop[:, 3, :, :] > 0
        if not mask.any():
            continue
        for ch_name, ch_idx in CHANNEL_INDICES.items():
            vals = crop[:, ch_idx, :, :][mask].astype(np.float64)
            pixel_pools[ch_name].append(vals)

    thresholds: dict[str, float] = {}
    for ch in CHANNEL_NAMES:
        if not pixel_pools[ch]:
            thresholds[ch] = 0.0
            continue
        all_vals = np.concatenate(pixel_pools[ch])
        pos = all_vals[all_vals > 0]
        if len(pos) < 2:
            thresholds[ch] = 0.0
            continue
        try:
            thresholds[ch] = float(np.exp(float(threshold_otsu(np.log(pos)))))
        except ValueError:
            thresholds[ch] = 0.0
    return thresholds


def run(output_dir: Path, method_str: str = "otsu", force: bool = False) -> int:
    qc_dir = output_dir / "cell_qc"
    input_csv = qc_dir / "qc_features.csv"
    output_csv = qc_dir / "qc_features_filtered.csv"
    if not input_csv.exists():
        print(f"ERROR: {input_csv} not found (run extract_features.py first)")
        return 1
    if output_csv.exists() and not force:
        print(f"SKIP (exists): {output_csv}  (use --force to overwrite)")
        return 0

    method, param = _parse_method(method_str)
    with open(str(input_csv), newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("No rows in input CSV.")
        return 1

    thresholds = {}
    for ch in CHANNEL_NAMES:
        values = np.array([float(r[f"mean_{ch}"]) for r in rows], dtype=np.float64)
        thresholds[ch] = _compute_threshold(values, method, param)

    box_dir = output_dir / "cell_boxing"
    px_thresholds = _compute_pixel_thresholds(box_dir)

    for row in rows:
        all_pass = True
        all_px_pass = True
        for ch in CHANNEL_NAMES:
            val = float(row[f"mean_{ch}"])
            passed = int(val >= thresholds[ch])
            row[f"threshold_{ch}"] = round(thresholds[ch], 4)
            row[f"pass_{ch}"] = passed
            if not passed:
                all_pass = False

            px_passed = int(val >= px_thresholds[ch])
            row[f"px_threshold_{ch}"] = round(px_thresholds[ch], 4)
            row[f"px_pass_{ch}"] = px_passed
            if not px_passed:
                all_px_pass = False
        row["pass_intensity"] = int(all_pass)
        row["pass_pixel_intensity"] = int(all_px_pass)

    new_cols = []
    for ch in CHANNEL_NAMES:
        new_cols.extend([f"threshold_{ch}", f"pass_{ch}", f"px_threshold_{ch}", f"px_pass_{ch}"])
    new_cols.extend(["pass_intensity", "pass_pixel_intensity"])
    original_cols = list(rows[0].keys())
    fieldnames = [c for c in original_cols if c not in new_cols] + new_cols

    qc_dir.mkdir(parents=True, exist_ok=True)
    tmp = str(output_csv) + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, str(output_csv))
    print(f"Wrote {output_csv}  ({len(rows)} cells, {len(fieldnames)} columns)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-rel", type=str, default=None, help="Relative path under data/ and output/")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--method", type=str, default="log_otsu")
    ap.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = ap.parse_args()
    _, output_dir = _resolve_dirs(args)
    return run(output_dir, method_str=args.method, force=args.force)


if __name__ == "__main__":
    sys.exit(main())

