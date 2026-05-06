#!/usr/bin/env python3
"""Export per-Z intensity stats for two warped 488 stacks (CSV for plotting).

Defaults compare Sample10_Position0 (488 issue) vs Sample9_Position3 (healthy 488).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile as tf


def z_series(path: Path):
    with tf.TiffFile(path) as tif:
        s = tif.series[0]
        z = int(s.shape[0])
        means = np.empty(z)
        p995 = np.empty(z)
        p999 = np.empty(z)
        vmax = np.empty(z)
        frac_500 = np.empty(z)
        frac_2000 = np.empty(z)
        for i in range(z):
            p = np.asarray(s.asarray(key=i), dtype=np.float64).ravel()
            means[i] = float(np.mean(p))
            p995[i] = float(np.percentile(p, 99.5))
            p999[i] = float(np.percentile(p, 99.9))
            vmax[i] = float(np.max(p))
            frac_500[i] = float(np.mean(p > 500))
            frac_2000[i] = float(np.mean(p > 2000))
    return {
        "Z": z,
        "mean": means,
        "p995": p995,
        "p999": p999,
        "max": vmax,
        "frac_gt_500": frac_500,
        "frac_gt_2000": frac_2000,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    scratch = Path(
        "/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysisScratch/"
        "Gutierrez/CMAP_cropped_copies/4_24_25_CGN_6_10_2"
    )
    ap.add_argument(
        "--bad",
        type=Path,
        default=scratch / "Sample10_Position0_decon_dsr" / "reg_488nm_to_642nm_fast_mi_Warped.tif",
        help="Warped 488 TIFF (problem sample)",
    )
    ap.add_argument(
        "--good",
        type=Path,
        default=scratch / "Sample9_Position3_decon_dsr" / "reg_488nm_to_642nm_fast_mi_Warped.tif",
        help="Warped 488 TIFF (reference sample)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: <repo>/logs/bad_vs_good_488_zstats.csv)",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out = args.out or (repo / "logs" / "bad_vs_good_488_zstats.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    bad = z_series(args.bad)
    good = z_series(args.good)
    z_use = min(bad["Z"], good["Z"])

    fieldnames = [
        "z",
        "bad_mean",
        "good_mean",
        "bad_p995",
        "good_p995",
        "bad_p999",
        "good_p999",
        "bad_max",
        "good_max",
        "bad_frac_gt_500",
        "good_frac_gt_500",
        "bad_frac_gt_2000",
        "good_frac_gt_2000",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for z in range(z_use):
            w.writerow(
                {
                    "z": z,
                    "bad_mean": bad["mean"][z],
                    "good_mean": good["mean"][z],
                    "bad_p995": bad["p995"][z],
                    "good_p995": good["p995"][z],
                    "bad_p999": bad["p999"][z],
                    "good_p999": good["p999"][z],
                    "bad_max": bad["max"][z],
                    "good_max": good["max"][z],
                    "bad_frac_gt_500": bad["frac_gt_500"][z],
                    "good_frac_gt_500": good["frac_gt_500"][z],
                    "bad_frac_gt_2000": bad["frac_gt_2000"][z],
                    "good_frac_gt_2000": good["frac_gt_2000"][z],
                }
            )

    print(f"Wrote {z_use} rows to {out}")
    print(f"  bad:  {args.bad}")
    print(f"  good: {args.good}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
