#!/usr/bin/env python3
"""One-pass sweep: Method A (488 positive-pool percentile trim) vs hybrid thresholds.

Loads union mask + combined once, then for each ``p_lo`` recomputes voxel
log-Otsu (with optional 488 trim), ``T_hybrid = max(log_otsu, bg_mean+X*bg_std)``
per channel, applies the same 488+560 intensity + shape gate as
``filter_by_intensity.py --from-volume --filter-channels 488,560``.

Writes a CSV table with per-row ``p_lo``, hybrid thresholds, raw voxel log-Otsu
(before max with bg), tracked cell pass on 488-only and full gate, and survivor
count.

Example (from repo root, heavy I/O — prefer ``bsub``):

  bsub -q standard -W 120 -M 32000 -R 'rusage[mem=32000]' \\
    python qc/sweep_voxel488_trim_methodA_report.py \\
      --sample-dir /path/to/CGNSample3_Position6_decon_dsr \\
      --track-cell-id 104 --output-csv /path/to/methodA_sweep_cell104.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# Import filter_by_intensity as a top-level module (qc/ is not a Python package).
sys.path.insert(0, str(_ROOT / "qc"))

from postprocess import variant_files  # noqa: E402

import filter_by_intensity as fbi  # noqa: E402

CHANNEL_INDICES = fbi.CHANNEL_INDICES
DEFAULT_SHAPE_PARAMS = fbi.DEFAULT_SHAPE_PARAMS
ShapeFilterParams = fbi.ShapeFilterParams
_bg_sigma_thresholds = fbi._bg_sigma_thresholds
_compute_background_stats = fbi._compute_background_stats
_log_otsu_per_channel = fbi._log_otsu_per_channel
_otsu_or_bg_thresholds = fbi._otsu_or_bg_thresholds
_pass_shape = fbi._pass_shape
_shape_metrics_3d = fbi._shape_metrics_3d


def _load_pass_shape_by_id(qc_csv: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    if not qc_csv.is_file():
        return out
    with open(qc_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                cid = int(float(row["cell_id"]))
                out[cid] = int(float(row.get("pass_shape", 1)))
            except (TypeError, ValueError, KeyError):
                continue
    return out


def run(
    sample_dir: Path,
    track_cell_id: int,
    x_bg: float,
    pctile_los: list[float],
    qc_csv: Path | None,
    shape_params: ShapeFilterParams,
    out_csv: Path,
) -> int:
    v = variant_files("union_488_560")
    mask_path = sample_dir / v["mask"]
    comb_path = sample_dir / v["combined"]
    if not mask_path.is_file() or not comb_path.is_file():
        print(f"ERROR: missing {mask_path} or {comb_path}")
        return 1

    mask = tifffile.imread(str(mask_path))
    if mask.ndim == 4:
        mask = mask[:, 0]
    mask_i = mask.astype(np.int32, copy=False)
    combined = tifffile.imread(str(comb_path))
    if combined.ndim != 4 or combined.shape[1] < 3:
        print(f"ERROR: expected ZCYX combined, got {combined.shape}")
        return 1
    z_min = min(mask_i.shape[0], combined.shape[0])
    mask_i = mask_i[:z_min]
    combined = combined[:z_min]

    label_ids = np.unique(mask_i)
    label_ids = label_ids[label_ids > 0]
    if label_ids.size == 0:
        print("ERROR: empty mask")
        return 1

    per_mean: dict[str, np.ndarray] = {}
    for ch_name, ch_idx in CHANNEL_INDICES.items():
        slab = combined[:, ch_idx, :, :].astype(np.float64, copy=False)
        per_mean[ch_name] = ndimage.mean(slab, labels=mask_i, index=label_ids)

    max_lab = int(mask_i.max())
    label_slices = ndimage.find_objects(mask_i, max_lab)
    pass_shape_by_id: dict[int, int] = {}
    for i, lid in enumerate(label_ids):
        cid = int(lid)
        if 0 < cid <= max_lab and label_slices[cid - 1] is not None:
            slc = label_slices[cid - 1]
            bin3d = mask_i[slc] == cid
        else:
            bin3d = mask_i == cid
        sm = _shape_metrics_3d(bin3d, erode_iterations=shape_params.erode_iterations)
        pass_shape_by_id[cid] = int(_pass_shape(sm, shape_params))

    ext = _load_pass_shape_by_id(qc_csv) if qc_csv else {}
    for cid, ps in ext.items():
        pass_shape_by_id[cid] = ps

    bg_stats = _compute_background_stats(combined, mask_i)
    bg_T = _bg_sigma_thresholds(bg_stats, x_bg)

    rows: list[dict[str, str | float | int]] = []
    for p_lo in pctile_los:
        log_T = _log_otsu_per_channel(combined, mask_i, voxel_488_positive_pctile_lo=float(p_lo))
        hybrid = _otsu_or_bg_thresholds(bg_stats, log_T, x_bg)
        filter_set = {"488", "560"}
        survivors = 0
        c_pass_488 = ""
        c_pass_560 = ""
        c_pass_full = ""
        c_mean_488 = ""
        c_mean_560 = ""
        for i, lid in enumerate(label_ids):
            cid = int(lid)
            ps = int(pass_shape_by_id.get(cid, 1))
            p488 = int(per_mean["488"][i] >= hybrid["488"])
            p560 = int(per_mean["560"][i] >= hybrid["560"])
            if p488 and p560 and ps:
                survivors += 1
            if cid == track_cell_id:
                c_mean_488 = round(float(per_mean["488"][i]), 6)
                c_mean_560 = round(float(per_mean["560"][i]), 6)
                c_pass_488 = p488
                c_pass_560 = p560
                c_pass_full = int(p488 and p560 and ps)

        rows.append(
            {
                "voxel_488_positive_pctile_lo": float(p_lo),
                "X_bg_sigma": float(x_bg),
                "log_otsu_voxel_488_raw": round(float(log_T["488"]), 6),
                "bg_sigma_T_488": round(float(bg_T["488"]), 6),
                "hybrid_T_488": round(float(hybrid["488"]), 6),
                "hybrid_T_560": round(float(hybrid["560"]), 6),
                "hybrid_T_642": round(float(hybrid["642"]), 6),
                f"track_cell_{track_cell_id}_mean_488": c_mean_488,
                f"track_cell_{track_cell_id}_mean_560": c_mean_560,
                f"track_cell_{track_cell_id}_pass_488_only": c_pass_488,
                f"track_cell_{track_cell_id}_pass_560_only": c_pass_560,
                f"track_cell_{track_cell_id}_pass_488560_shape_gate": c_pass_full,
                "n_cells": int(label_ids.size),
                "n_survivors_after_488560_shape": int(survivors),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}  ({len(rows)} rows)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--track-cell-id", type=int, default=104)
    ap.add_argument("--x-bg-sigma", type=float, default=3.0)
    ap.add_argument(
        "--pctile-los",
        type=str,
        default="0,0.5,1,1.5,2,3,5",
        help="Comma-separated voxel_488_positive_pctile_lo sweep values",
    )
    ap.add_argument(
        "--qc-csv",
        type=Path,
        default=None,
        help="Optional qc_features_filtered.csv to override pass_shape per cell_id",
    )
    ap.add_argument("--shape-filter", choices=("on", "off"), default="on")
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    sp = DEFAULT_SHAPE_PARAMS
    if args.shape_filter == "off":
        sp = ShapeFilterParams(
            enabled=False,
            max_bbox_aspect=sp.max_bbox_aspect,
            min_fill_ratio=sp.min_fill_ratio,
            max_inertia_ratio=sp.max_inertia_ratio,
            min_vol_erode_test=sp.min_vol_erode_test,
            min_vol_skip_all=sp.min_vol_skip_all,
            erode_iterations=sp.erode_iterations,
        )

    qc = args.qc_csv
    if qc is None:
        qc = args.sample_dir / "cell_qc_union_488_560" / "qc_features_filtered.csv"

    los = [float(x.strip()) for x in args.pctile_los.split(",") if x.strip()]
    return run(
        args.sample_dir.resolve(),
        args.track_cell_id,
        args.x_bg_sigma,
        los,
        qc.resolve() if qc else None,
        sp,
        args.output_csv.resolve(),
    )


if __name__ == "__main__":
    sys.exit(main())
