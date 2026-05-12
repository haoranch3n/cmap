#!/usr/bin/env python3
"""Robust per-cell gate (642/488/560): bg median + k·MAD, voxel_thr, q75 + pos_frac.

Per channel independently (same bg mask as production dilated complement):
  sigma_eff = max(MAD_SCALE * MAD(bg), noise_floor_ch)
  voxel_thr_ch = max(bg_median_ch + k * sigma_eff_ch, intensity_floor_ch)

Per cell:
  q_ch = percentile(cell voxels, q_pct)
  pos_frac_ch = fraction of cell voxels with value > voxel_thr_ch

Pass channel if: q_ch > voxel_thr_ch AND pos_frac_ch > f_min

Pass intensity: all three channels pass.

Optional AND pass_shape from cell_qc_union_488_560/qc_features_filtered.csv.

Output: one CSV row per (k, f_min) with thresholds, cell 104 pass/fail per channel,
fail_channels string, survivor counts.
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
sys.path.insert(0, str(_ROOT / "qc"))

import filter_by_intensity as fbi  # noqa: E402

from postprocess import variant_files  # noqa: E402

CHANNEL_NAMES = fbi.CHANNEL_NAMES
CHANNEL_INDICES = fbi.CHANNEL_INDICES
MAD_SCALE = 1.4826
BG_SAMPLE_LIMIT = 10_000_000


def _bg_median_mad_channel(
    bg_vals: np.ndarray, top_clip_pct: float
) -> tuple[float, float, int]:
    if bg_vals.size == 0:
        return 0.0, 0.0, 0
    if top_clip_pct > 0:
        cutoff = float(np.percentile(bg_vals, 100.0 - top_clip_pct))
        bg_kept = bg_vals[bg_vals <= cutoff]
    else:
        bg_kept = bg_vals
    if bg_kept.size == 0:
        return 0.0, 0.0, 0
    if bg_kept.size > BG_SAMPLE_LIMIT:
        rng = np.random.default_rng(seed=0)
        sub_idx = rng.choice(bg_kept.size, size=BG_SAMPLE_LIMIT, replace=False)
        sub = bg_kept[sub_idx]
    else:
        sub = bg_kept
    med = float(np.median(sub))
    mad = float(np.median(np.abs(sub - med))) if sub.size else 0.0
    return med, mad, int(bg_kept.size)


def _bg_mask(mask_i: np.ndarray, dilate_iters: int) -> np.ndarray:
    fg = mask_i > 0
    if dilate_iters > 0:
        struct = ndimage.generate_binary_structure(3, 1)
        fg_dilated = ndimage.binary_dilation(fg, structure=struct, iterations=dilate_iters)
    else:
        fg_dilated = fg
    return ~fg_dilated


def _load_pass_shape(qc_csv: Path) -> dict[int, int]:
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


def _per_cell_volume(mask_i: np.ndarray, label_ids: np.ndarray) -> np.ndarray:
    ones = np.ones_like(mask_i, dtype=np.float32)
    return np.asarray(ndimage.sum(ones, labels=mask_i, index=label_ids), dtype=np.float64)


def _per_cell_q_percentile(
    slab: np.ndarray, mask_i: np.ndarray, label_ids: np.ndarray, q_pct: float
) -> np.ndarray:
    out = np.zeros(label_ids.size, dtype=np.float64)
    max_lab = int(mask_i.max())
    slices = ndimage.find_objects(mask_i, max_lab)
    for i, lid in enumerate(label_ids):
        lid_i = int(lid)
        if not (0 < lid_i <= max_lab) or slices[lid_i - 1] is None:
            continue
        slc = slices[lid_i - 1]
        sub = mask_i[slc] == lid_i
        if not sub.any():
            continue
        vals = slab[slc][sub]
        if vals.size:
            out[i] = float(np.percentile(vals, q_pct))
    return out


def _per_cell_frac_above_thr(
    slab: np.ndarray, mask_i: np.ndarray, label_ids: np.ndarray, thr: float, vol: np.ndarray
) -> np.ndarray:
    above = (slab > thr).astype(np.float32, copy=False)
    sum_a = np.asarray(ndimage.sum(above, labels=mask_i, index=label_ids), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vol > 0, sum_a / np.maximum(vol, 1.0), 0.0)


def run_once(
    *,
    mask_i: np.ndarray,
    combined: np.ndarray,
    label_ids: np.ndarray,
    bg_mask: np.ndarray,
    pass_shape_by_id: dict[int, int],
    track_id: int,
    k: float,
    f_min: float,
    q_pct: float,
    noise_floor: dict[str, float],
    intensity_floor: dict[str, float],
    top_clip_pct: float,
) -> dict[str, float | int | str]:
    bg_med: dict[str, float] = {}
    bg_mad: dict[str, float] = {}
    voxel_thr: dict[str, float] = {}
    for ch in CHANNEL_NAMES:
        ch_idx = CHANNEL_INDICES[ch]
        bg_vals = combined[:, ch_idx, :, :][bg_mask].astype(np.float64, copy=False)
        med, mad, _n = _bg_median_mad_channel(bg_vals, top_clip_pct)
        bg_med[ch] = med
        bg_mad[ch] = mad
        sigma_eff = max(MAD_SCALE * mad, float(noise_floor.get(ch, 0.0)))
        raw_thr = med + k * sigma_eff
        voxel_thr[ch] = max(raw_thr, float(intensity_floor.get(ch, 0.0)))

    vol = _per_cell_volume(mask_i, label_ids)
    pass_ch: dict[str, np.ndarray] = {}
    q_store: dict[str, np.ndarray] = {}
    frac_store: dict[str, np.ndarray] = {}
    for ch in CHANNEL_NAMES:
        ch_idx = CHANNEL_INDICES[ch]
        slab = combined[:, ch_idx, :, :].astype(np.float64, copy=False)
        q_store[ch] = _per_cell_q_percentile(slab, mask_i, label_ids, q_pct)
        frac_store[ch] = _per_cell_frac_above_thr(slab, mask_i, label_ids, voxel_thr[ch], vol)
        pass_ch[ch] = (q_store[ch] > voxel_thr[ch]) & (frac_store[ch] > f_min)

    pass_all = pass_ch["642"] & pass_ch["488"] & pass_ch["560"]
    survivors_i = int(pass_all.sum())

    shape_arr = np.array(
        [bool(pass_shape_by_id.get(int(lid), 1)) for lid in label_ids], dtype=bool
    )
    pass_all_shape = pass_all & shape_arr
    survivors_shape = int(pass_all_shape.sum())

    idx104: int | None = None
    for i, lid in enumerate(label_ids):
        if int(lid) == track_id:
            idx104 = i
            break

    if idx104 is None:
        r104 = {
            "pass_642": "",
            "pass_488": "",
            "pass_560": "",
            "fail_channels": "not_found",
            "pass_all_intensity": "",
            "pass_all_intensity_shape": "",
            "q75_642": "",
            "q75_488": "",
            "q75_560": "",
            "pos_frac_642": "",
            "pos_frac_488": "",
            "pos_frac_560": "",
        }
    else:
        i = idx104
        fails = [ch for ch in CHANNEL_NAMES if not bool(pass_ch[ch][i])]
        r104 = {
            "pass_642": int(bool(pass_ch["642"][i])),
            "pass_488": int(bool(pass_ch["488"][i])),
            "pass_560": int(bool(pass_ch["560"][i])),
            "fail_channels": "+".join(fails) if fails else "",
            "pass_all_intensity": int(bool(pass_all[i])),
            "pass_all_intensity_shape": int(bool(pass_all_shape[i])),
            "q75_642": round(float(q_store["642"][i]), 6),
            "q75_488": round(float(q_store["488"][i]), 6),
            "q75_560": round(float(q_store["560"][i]), 6),
            "pos_frac_642": round(float(frac_store["642"][i]), 6),
            "pos_frac_488": round(float(frac_store["488"][i]), 6),
            "pos_frac_560": round(float(frac_store["560"][i]), 6),
        }

    row: dict[str, float | int | str] = {
        "k": k,
        "f_min": f_min,
        "q_pct": q_pct,
        "voxel_thr_642": round(voxel_thr["642"], 6),
        "voxel_thr_488": round(voxel_thr["488"], 6),
        "voxel_thr_560": round(voxel_thr["560"], 6),
        "bg_median_642": round(bg_med["642"], 6),
        "bg_median_488": round(bg_med["488"], 6),
        "bg_median_560": round(bg_med["560"], 6),
        "bg_mad_642": round(bg_mad["642"], 6),
        "bg_mad_488": round(bg_mad["488"], 6),
        "bg_mad_560": round(bg_mad["560"], 6),
        "n_cells": int(label_ids.size),
        "n_survivors_all3": survivors_i,
        "n_survivors_all3_and_shape": survivors_shape,
    }
    tid = int(track_id)
    for kk, vv in r104.items():
        row[f"track_{tid}_{kk}"] = vv
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--track-cell-id", type=int, default=104)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--variant", type=str, default="union_488_560")
    ap.add_argument("--bg-dilate-iters", type=int, default=2)
    ap.add_argument("--bg-top-clip-pct", type=float, default=1.0)
    ap.add_argument("--q-pct", type=float, default=75.0)
    ap.add_argument("--k-list", type=str, default="3,4,5")
    ap.add_argument("--f-min-list", type=str, default="0.05,0.10")
    ap.add_argument("--noise-floor", type=str, default="0,0,0", help="642,488,560")
    ap.add_argument("--intensity-floor", type=str, default="0,0,0", help="642,488,560")
    ap.add_argument("--qc-csv", type=Path, default=None)
    args = ap.parse_args()

    sample_dir = args.sample_dir.resolve()
    v = variant_files(args.variant)
    mask_path = sample_dir / v["mask"]
    comb_path = sample_dir / v["combined"]
    if not mask_path.is_file() or not comb_path.is_file():
        print(f"ERROR: missing {mask_path} or {comb_path}")
        return 1

    qc = args.qc_csv or (sample_dir / "cell_qc_union_488_560" / "qc_features_filtered.csv")
    pass_shape_by_id = _load_pass_shape(qc.resolve())

    mask = tifffile.imread(str(mask_path))
    if mask.ndim == 4:
        mask = mask[:, 0]
    mask_i = mask.astype(np.int32, copy=False)
    combined = tifffile.imread(str(comb_path))
    if combined.ndim != 4 or combined.shape[1] < 3:
        print(f"ERROR: bad combined shape {combined.shape}")
        return 1
    z_min = min(mask_i.shape[0], combined.shape[0])
    mask_i = mask_i[:z_min]
    combined = combined[:z_min]

    label_ids = np.unique(mask_i)
    label_ids = label_ids[label_ids > 0]
    if label_ids.size == 0:
        print("ERROR: no labels")
        return 1

    bg_mask_arr = _bg_mask(mask_i, max(0, int(args.bg_dilate_iters)))

    nf = [float(x) for x in args.noise_floor.split(",")]
    inf = [float(x) for x in args.intensity_floor.split(",")]
    if len(nf) != 3 or len(inf) != 3:
        print("ERROR: --noise-floor and --intensity-floor need three values")
        return 1
    noise_floor = {ch: nf[i] for i, ch in enumerate(CHANNEL_NAMES)}
    intensity_floor = {ch: inf[i] for i, ch in enumerate(CHANNEL_NAMES)}

    ks = [float(x) for x in args.k_list.split(",") if x.strip()]
    fms = [float(x) for x in args.f_min_list.split(",") if x.strip()]
    q_pct = float(args.q_pct)

    rows: list[dict[str, float | int | str]] = []
    for k in ks:
        for f_min in fms:
            rows.append(
                run_once(
                    mask_i=mask_i,
                    combined=combined,
                    label_ids=label_ids,
                    bg_mask=bg_mask_arr,
                    pass_shape_by_id=pass_shape_by_id,
                    track_id=int(args.track_cell_id),
                    k=k,
                    f_min=f_min,
                    q_pct=q_pct,
                    noise_floor=noise_floor,
                    intensity_floor=intensity_floor,
                    top_clip_pct=max(0.0, float(args.bg_top_clip_pct)),
                )
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.output_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output_csv}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
