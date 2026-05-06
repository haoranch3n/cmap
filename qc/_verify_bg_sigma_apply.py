#!/usr/bin/env python3
"""Verify a bg_sigma-filtered run.

Reads the QC CSV and the filtered output TIFF, and reports:
- thresholds per channel,
- total pass count,
- whether ``--track-cell`` survives in both the CSV pass column and the TIFF
  (must be zeroed-out in the TIFF if pass=0).

Used as a Phase 3 verification step from ``qc/bsub_bg_sigma_apply.sh``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--output-mask", type=Path, required=True)
    ap.add_argument("--pass-column", type=str, default="pass_bg_sigma_shape")
    ap.add_argument("--track-cell", type=int, default=25)
    args = ap.parse_args()

    if not args.csv.is_file():
        print(f"ERROR: CSV missing: {args.csv}")
        return 1
    if not args.output_mask.is_file():
        print(f"ERROR: output mask missing: {args.output_mask}")
        return 1

    with open(args.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("ERROR: empty CSV")
        return 1
    if args.pass_column not in rows[0]:
        print(f"ERROR: column {args.pass_column!r} not in CSV; have: {sorted(rows[0].keys())}")
        return 1

    n_total = len(rows)
    n_pass = sum(1 for r in rows if int(float(r[args.pass_column])) == 1)

    r0 = rows[0]
    print(f"  CSV: {args.csv}")
    print(f"  rows={n_total}  {args.pass_column}={n_pass}/{n_total}")
    for ch in ("642", "488", "560"):
        if f"threshold_{ch}" in r0:
            print(f"  threshold_{ch}={r0[f'threshold_{ch}']}  px_threshold_{ch}={r0.get(f'px_threshold_{ch}', '-')}")

    tracked: dict | None = None
    for r in rows:
        try:
            if int(float(r["cell_id"])) == args.track_cell:
                tracked = r
                break
        except (ValueError, KeyError):
            continue

    if tracked is None:
        print(f"  track_cell={args.track_cell}: not present in CSV")
    else:
        cell_pass = int(float(tracked[args.pass_column]))
        print(
            f"  track_cell={args.track_cell}: "
            f"mean_642={tracked.get('mean_642','-')}  "
            f"mean_488={tracked.get('mean_488','-')}  "
            f"mean_560={tracked.get('mean_560','-')}  "
            f"{args.pass_column}={cell_pass}"
        )

    print(f"  loading mask: {args.output_mask}")
    mask = tifffile.imread(str(args.output_mask))
    n_label_in_mask = int(np.unique(mask[mask > 0]).size)
    print(f"  filtered mask: {n_label_in_mask} non-zero labels")

    track_in_mask = bool(np.any(mask == args.track_cell))
    print(f"  track_cell={args.track_cell} present in filtered mask: {track_in_mask}")

    expected_in_mask = (tracked is not None and int(float(tracked[args.pass_column])) == 1)
    if track_in_mask != expected_in_mask:
        print(
            f"  WARN: filtered mask vs CSV disagree for cell {args.track_cell} "
            f"(CSV expected_in_mask={expected_in_mask}, mask says present={track_in_mask})"
        )
    else:
        print(
            f"  OK: cell {args.track_cell} CSV+mask agree "
            f"({'kept' if track_in_mask else 'removed'})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
