#!/usr/bin/env python3
"""Print sample relpaths (under output/) ready for DINOv2 extract LSF batch.

One line per sample: e.g. ``4_18_25/CGNSample1_Position0_decon_dsr``

Criteria: ``dinov2_volume_norm_bounds.csv`` exists, cell boxing dir has >=1
``cell_*.tif``, skip samples that already have ``dinov2_embeddings.csv`` in
the given QC dir unless ``--include-done``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def default_cell_qc_dir(cell_boxing_dir: str) -> str:
    if cell_boxing_dir == "cell_boxing_filtered":
        return "cell_qc_filtered"
    return "cell_qc"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "dataset",
        type=str,
        help="Folder name under output/, e.g. 4_18_25 or 4_24_25_CGN_6_10_2",
    )
    ap.add_argument(
        "--cell-boxing-dir",
        type=str,
        default="cell_boxing",
        help="Subfolder with cell_*.tif (default: cell_boxing)",
    )
    ap.add_argument(
        "--cell-qc-dir",
        type=str,
        default=None,
        help="QC/embeddings subfolder (default: cell_qc or cell_qc_filtered if boxing filtered)",
    )
    ap.add_argument(
        "--include-done",
        action="store_true",
        help="Also list samples that already have dinov2_embeddings.csv",
    )
    args = ap.parse_args()

    cell_qc_dir = args.cell_qc_dir or default_cell_qc_dir(args.cell_boxing_dir)

    cmap_root = Path(os.environ.get("CMAP_ROOT", Path(__file__).resolve().parents[2]))
    out_root = cmap_root / "output" / args.dataset
    if not out_root.is_dir():
        print(f"ERROR: not a directory: {out_root}", flush=True)
        return 1

    list_prefix = Path(args.dataset)
    n = 0
    for p in sorted(out_root.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        box = p / args.cell_boxing_dir
        bounds = p / "dinov2_volume_norm_bounds.csv"
        done_csv = p / cell_qc_dir / "dinov2_embeddings.csv"
        if not bounds.is_file():
            continue
        if not box.is_dir():
            continue
        if not any(box.glob("cell_*.tif")):
            continue
        if done_csv.is_file() and not args.include_done:
            continue
        print(list_prefix / p.name, flush=True)
        n += 1

    print(f"# listed {n} samples", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
