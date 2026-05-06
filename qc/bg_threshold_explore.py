#!/usr/bin/env python3
"""
Background-statistics threshold exploration for per-channel intensity QC.

Evaluates 5 candidate filtering methods plus 2 baselines on a sample's mask +
combined TIFF, sweeping X over {3, 5, 7, 10}. Reports per-method/X thresholds,
total pass counts, and explicitly tracks whether ``--track-cell`` (default 25)
passes under each setting.

Methods (per channel, all on optionally-clipped voxels):
- M1 ``bg_sigma``      : T = bg_mean + X * bg_std                    (manager's idea)
- M2 ``bg_mad``        : T = bg_median + X * 1.4826 * MAD            (robust scale)
- M3 ``otsu_or_bg``    : T = max(log_otsu, bg_mean + X * bg_std)     (hybrid floor)
- M4 ``bg_frac``       : pass if frac_voxels_above(bg_mean+X*std) >= min_frac
- M5 ``bg_pct95``      : pass if pct95_cell >= bg_mean + X * bg_std

Baselines (X-independent, repeated for each row):
- ``log_otsu``         : T = exp(threshold_otsu(log(values>0)))
- ``otsu2``            : T = threshold_otsu(clip(values, 0, p99.99))

Background-stat refinements (always reported, both ON):
- Dilate the cell mask by N voxels (default 2) before taking complement.
- Drop top P% of background voxels (default 1%) as autofluorescent debris.

Negative voxels are treated as deconvolution artifacts and clipped to 0
(per manager's note); pass --no-clip-negatives to disable for sensitivity.

Output:
- ``<sample>/bg_threshold_report_<variant>.csv`` with one row per
  (method, X, min_frac).
- ``<sample>/bg_threshold_report_<variant>_track_cell_detail.csv`` with the
  tracked cell's per-channel raw stats.
- A console table summarising n_pass and the cell25 flag.

Usage:
    python qc/bg_threshold_explore.py \
        --output-dir output/4_24_25_CGN_6_10_2/Sample7_Position7_decon_dsr \
        --variant union_488_560 \
        --track-cell 25
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
from scipy import ndimage

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from postprocess import DEFAULT_VARIANT, VALID_VARIANTS, variant_files  # noqa: E402

CHANNEL_NAMES = ["642", "488", "560"]
CHANNEL_INDICES = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}

X_SWEEP = (3.0, 5.0, 7.0, 10.0)
MIN_FRAC_SWEEP = (0.10, 0.25)
BG_SAMPLE_LIMIT = 10_000_000  # cap subsample for MAD / percentile


def _bg_stats(bg_vals: np.ndarray, top_clip_pct: float) -> dict[str, float]:
    """Per-channel background descriptors, optionally dropping top P%.

    Mean/std use the full sample; median/MAD use a capped subsample because
    exact median on hundreds of millions of values is wasteful.
    """
    if bg_vals.size == 0:
        return {"bg_mean": 0.0, "bg_std": 0.0, "bg_median": 0.0, "bg_mad": 0.0, "bg_n": 0}

    if top_clip_pct > 0:
        cutoff = float(np.percentile(bg_vals, 100.0 - top_clip_pct))
        bg_kept = bg_vals[bg_vals <= cutoff]
    else:
        bg_kept = bg_vals

    bg_mean = float(np.mean(bg_kept)) if bg_kept.size else 0.0
    bg_std = float(np.std(bg_kept)) if bg_kept.size else 0.0

    if bg_kept.size > BG_SAMPLE_LIMIT:
        rng = np.random.default_rng(seed=0)
        sub_idx = rng.choice(bg_kept.size, size=BG_SAMPLE_LIMIT, replace=False)
        sub = bg_kept[sub_idx]
    else:
        sub = bg_kept
    bg_median = float(np.median(sub)) if sub.size else 0.0
    bg_mad = float(np.median(np.abs(sub - bg_median))) if sub.size else 0.0
    return {"bg_mean": bg_mean, "bg_std": bg_std, "bg_median": bg_median, "bg_mad": bg_mad, "bg_n": int(bg_kept.size)}


def _otsu_log(values: np.ndarray) -> float:
    from skimage.filters import threshold_otsu
    pos = values[values > 0]
    if pos.size < 2:
        return 0.0
    try:
        return float(np.exp(float(threshold_otsu(np.log(pos.astype(np.float64))))))
    except ValueError:
        return 0.0


def _otsu2(values: np.ndarray) -> float:
    from skimage.filters import threshold_otsu
    if values.size < 2:
        return 0.0
    p99 = float(np.percentile(values, 99.99))
    clipped = np.clip(values, 0, p99)
    try:
        return float(threshold_otsu(clipped))
    except ValueError:
        return 0.0


def _per_cell_pct95(slab: np.ndarray, mask_i: np.ndarray, label_ids: np.ndarray) -> np.ndarray:
    pct95 = np.zeros(label_ids.size, dtype=np.float64)
    max_lab = int(mask_i.max())
    if max_lab < 1:
        return pct95
    slices = ndimage.find_objects(mask_i, max_lab)
    for i, lid in enumerate(label_ids):
        slc = slices[lid - 1] if 0 < lid <= max_lab else None
        if slc is None:
            continue
        sub_mask = mask_i[slc] == lid
        if not sub_mask.any():
            continue
        vals = slab[slc][sub_mask]
        if vals.size:
            pct95[i] = float(np.percentile(vals, 95))
    return pct95


def _per_cell_frac_above(
    slab: np.ndarray, mask_i: np.ndarray, label_ids: np.ndarray, T: float, vol: np.ndarray
) -> np.ndarray:
    """Fraction of cell voxels with intensity >= T (per cell)."""
    above = (slab >= T).astype(np.float32, copy=False)
    sum_above = np.asarray(ndimage.sum(above, labels=mask_i, index=label_ids), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(vol > 0, sum_above / np.maximum(vol, 1), 0.0)
    return frac


def _evaluate_methods(
    per_ch_mean: dict[str, np.ndarray],
    per_ch_pct95: dict[str, np.ndarray],
    per_ch_frac_above: dict[str, dict[float, np.ndarray]],  # keyed [ch][X]
    bg: dict[str, dict[str, float]],
    otsu_log_T: dict[str, float],
    otsu2_T: dict[str, float],
    label_ids: np.ndarray,
    track_cell: int | None,
) -> list[dict[str, float | int | str]]:
    """Build a list of result rows, one per (method, X, min_frac) combination."""
    n_cells = label_ids.size
    cell_idx_for_track = None
    if track_cell is not None:
        match = np.where(label_ids == track_cell)[0]
        if match.size:
            cell_idx_for_track = int(match[0])

    rows: list[dict[str, float | int | str]] = []

    def _emit(method: str, X: float, min_frac: float, T: dict[str, float], pass_per_ch: dict[str, np.ndarray]) -> None:
        all_pass = np.ones(n_cells, dtype=bool)
        for ch in CHANNEL_NAMES:
            all_pass &= pass_per_ch[ch]
        n_pass = int(all_pass.sum())
        row: dict[str, float | int | str] = {
            "method": method,
            "X": float(X),
            "min_frac": float(min_frac),
            "n_cells": int(n_cells),
            "n_pass_all3": n_pass,
        }
        if cell_idx_for_track is not None:
            row["track_cell_id"] = int(track_cell)
            row["track_cell_pass_all3"] = int(bool(all_pass[cell_idx_for_track]))
            for ch in CHANNEL_NAMES:
                row[f"track_cell_pass_{ch}"] = int(bool(pass_per_ch[ch][cell_idx_for_track]))
        for ch in CHANNEL_NAMES:
            row[f"T_{ch}"] = float(T[ch])
            row[f"n_pass_{ch}"] = int(pass_per_ch[ch].sum())
        rows.append(row)

    for X in X_SWEEP:
        T_sigma = {ch: bg[ch]["bg_mean"] + X * bg[ch]["bg_std"] for ch in CHANNEL_NAMES}
        T_mad = {ch: bg[ch]["bg_median"] + X * 1.4826 * bg[ch]["bg_mad"] for ch in CHANNEL_NAMES}
        T_hybrid = {ch: max(otsu_log_T[ch], T_sigma[ch]) for ch in CHANNEL_NAMES}

        pass_per_ch = {ch: per_ch_mean[ch] >= T_sigma[ch] for ch in CHANNEL_NAMES}
        _emit("bg_sigma", X, 0.0, T_sigma, pass_per_ch)

        pass_per_ch = {ch: per_ch_mean[ch] >= T_mad[ch] for ch in CHANNEL_NAMES}
        _emit("bg_mad", X, 0.0, T_mad, pass_per_ch)

        pass_per_ch = {ch: per_ch_mean[ch] >= T_hybrid[ch] for ch in CHANNEL_NAMES}
        _emit("otsu_or_bg", X, 0.0, T_hybrid, pass_per_ch)

        pass_per_ch = {ch: per_ch_pct95[ch] >= T_sigma[ch] for ch in CHANNEL_NAMES}
        _emit("bg_pct95", X, 0.0, T_sigma, pass_per_ch)

        for min_frac in MIN_FRAC_SWEEP:
            pass_per_ch = {ch: per_ch_frac_above[ch][X] >= min_frac for ch in CHANNEL_NAMES}
            _emit("bg_frac", X, min_frac, T_sigma, pass_per_ch)

    pass_per_ch = {ch: per_ch_mean[ch] >= otsu_log_T[ch] for ch in CHANNEL_NAMES}
    _emit("log_otsu", float("nan"), 0.0, otsu_log_T, pass_per_ch)

    pass_per_ch = {ch: per_ch_mean[ch] >= otsu2_T[ch] for ch in CHANNEL_NAMES}
    _emit("otsu2", float("nan"), 0.0, otsu2_T, pass_per_ch)

    return rows


def run(
    sample_dir: Path,
    variant: str,
    track_cell: int | None,
    clip_negatives: bool,
    bg_dilate_iters: int,
    bg_top_clip_pct: float,
    report_csv: Path | None,
) -> int:
    v = variant_files(variant)
    mask_path = sample_dir / v["mask"]
    combined_path = sample_dir / v["combined"]
    if not mask_path.is_file():
        print(f"ERROR: mask not found: {mask_path}")
        return 1
    if not combined_path.is_file():
        print(f"ERROR: combined not found: {combined_path}")
        return 1

    print(f"Sample: {sample_dir}")
    print(f"Variant: {variant}  mask={v['mask']}  combined={v['combined']}")
    print(f"clip_negatives={clip_negatives}  bg_dilate_iters={bg_dilate_iters}  bg_top_clip_pct={bg_top_clip_pct}")

    mask = tifffile.imread(str(mask_path))
    if mask.ndim == 4:
        mask = mask[:, 0]
    if mask.ndim != 3:
        print(f"ERROR: expected 3D mask, got {mask.shape}")
        return 1
    mask_i = mask.astype(np.int32, copy=False)

    combined = tifffile.imread(str(combined_path))
    if combined.ndim != 4 or combined.shape[1] < 3:
        print(f"ERROR: expected (Z, C>=3, Y, X) combined, got {combined.shape}")
        return 1
    z_min = min(mask_i.shape[0], combined.shape[0])
    mask_i = mask_i[:z_min]
    combined = combined[:z_min]
    print(f"Volume: mask={mask_i.shape}  combined={combined.shape}  dtype={combined.dtype}")

    label_ids = np.unique(mask_i)
    label_ids = label_ids[label_ids > 0]
    if label_ids.size == 0:
        print("ERROR: mask has no positive labels")
        return 1
    print(f"Labels: {label_ids.size} (max id = {int(label_ids.max())})")
    track_present = track_cell is not None and (track_cell in label_ids)
    if track_cell is not None and not track_present:
        print(f"WARNING: track_cell={track_cell} not present in mask (will be omitted from per-cell detail)")

    fg = mask_i > 0
    if bg_dilate_iters > 0:
        struct = ndimage.generate_binary_structure(3, 1)
        fg_dilated = ndimage.binary_dilation(fg, structure=struct, iterations=bg_dilate_iters)
    else:
        fg_dilated = fg
    bg_mask = ~fg_dilated
    print(f"Background voxels: {int(bg_mask.sum()):,d} / {int(bg_mask.size):,d}  ({100*bg_mask.mean():.2f}%)")

    vol_per_cell = np.asarray(
        ndimage.sum((mask_i > 0).astype(np.float32), labels=mask_i, index=label_ids),
        dtype=np.float64,
    )

    bg: dict[str, dict[str, float]] = {}
    per_ch_mean: dict[str, np.ndarray] = {}
    per_ch_pct95: dict[str, np.ndarray] = {}
    per_ch_frac_above: dict[str, dict[float, np.ndarray]] = {}
    otsu_log_T: dict[str, float] = {}
    otsu2_T: dict[str, float] = {}

    for ch in CHANNEL_NAMES:
        ch_idx = CHANNEL_INDICES[ch]
        slab_view = combined[:, ch_idx, :, :]
        if slab_view.dtype != np.float32:
            slab = slab_view.astype(np.float32)
        else:
            slab = slab_view
        if clip_negatives:
            np.maximum(slab, 0.0, out=slab)

        bg_vals = slab[bg_mask]
        bg[ch] = _bg_stats(bg_vals, top_clip_pct=bg_top_clip_pct)
        del bg_vals

        means = np.asarray(ndimage.mean(slab, labels=mask_i, index=label_ids), dtype=np.float64)
        per_ch_mean[ch] = means
        otsu_log_T[ch] = _otsu_log(means)
        otsu2_T[ch] = _otsu2(means)

        per_ch_pct95[ch] = _per_cell_pct95(slab, mask_i, label_ids)

        frac_for_X: dict[float, np.ndarray] = {}
        for X in X_SWEEP:
            T = bg[ch]["bg_mean"] + X * bg[ch]["bg_std"]
            frac_for_X[X] = _per_cell_frac_above(slab, mask_i, label_ids, T, vol_per_cell)
        per_ch_frac_above[ch] = frac_for_X

        print(
            f"  ch {ch}: bg_mean={bg[ch]['bg_mean']:.4g}  bg_std={bg[ch]['bg_std']:.4g}  "
            f"bg_median={bg[ch]['bg_median']:.4g}  bg_mad={bg[ch]['bg_mad']:.4g}  "
            f"bg_n={bg[ch]['bg_n']:,d}  "
            f"otsu_log_T={otsu_log_T[ch]:.4g}  otsu2_T={otsu2_T[ch]:.4g}"
        )

    rows = _evaluate_methods(
        per_ch_mean=per_ch_mean,
        per_ch_pct95=per_ch_pct95,
        per_ch_frac_above=per_ch_frac_above,
        bg=bg,
        otsu_log_T=otsu_log_T,
        otsu2_T=otsu2_T,
        label_ids=label_ids,
        track_cell=track_cell,
    )

    print()
    print(f"=== Method comparison (track_cell={track_cell}, present={track_present}) ===")
    hdr = (
        f"{'method':<14} {'X':>5} {'min_frac':>9} "
        f"{'T_642':>10} {'T_488':>10} {'T_560':>10} "
        f"{'n_pass':>8} {'tc_pass':>9} {'tc_642':>8} {'tc_488':>8} {'tc_560':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        x_val = r["X"]
        x_str = f"{x_val:.0f}" if not (isinstance(x_val, float) and np.isnan(x_val)) else "  -"
        tc_pass = r.get("track_cell_pass_all3", "-")
        tc_642 = r.get("track_cell_pass_642", "-")
        tc_488 = r.get("track_cell_pass_488", "-")
        tc_560 = r.get("track_cell_pass_560", "-")
        print(
            f"{r['method']:<14} {x_str:>5} {r['min_frac']:>9.2f} "
            f"{r['T_642']:>10.4g} {r['T_488']:>10.4g} {r['T_560']:>10.4g} "
            f"{r['n_pass_all3']:>8d} {str(tc_pass):>9} {str(tc_642):>8} {str(tc_488):>8} {str(tc_560):>8}"
        )

    if report_csv is None:
        report_csv = sample_dir / f"bg_threshold_report_{variant}.csv"
    fieldnames: list[str] = ["method", "X", "min_frac", "n_cells", "n_pass_all3"]
    if track_cell is not None:
        fieldnames += ["track_cell_id", "track_cell_pass_all3"]
        for ch in CHANNEL_NAMES:
            fieldnames.append(f"track_cell_pass_{ch}")
    for ch in CHANNEL_NAMES:
        fieldnames += [f"T_{ch}", f"n_pass_{ch}"]
    fieldnames.extend(["clip_negatives", "bg_dilate_iters", "bg_top_clip_pct"])
    for ch in CHANNEL_NAMES:
        for k in ("bg_mean", "bg_std", "bg_median", "bg_mad", "bg_n"):
            fieldnames.append(f"{k}_{ch}")

    for r in rows:
        r["clip_negatives"] = int(clip_negatives)
        r["bg_dilate_iters"] = int(bg_dilate_iters)
        r["bg_top_clip_pct"] = float(bg_top_clip_pct)
        for ch in CHANNEL_NAMES:
            for k in ("bg_mean", "bg_std", "bg_median", "bg_mad", "bg_n"):
                r[f"{k}_{ch}"] = bg[ch][k]

    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(report_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote report: {report_csv}  ({len(rows)} rows)")

    if track_present:
        per_cell_csv = report_csv.with_name(report_csv.stem + "_track_cell_detail.csv")
        idx = int(np.where(label_ids == track_cell)[0][0])
        with open(per_cell_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["channel", "mean", "pct95", "volume_voxels"])
            for ch in CHANNEL_NAMES:
                w.writerow([
                    ch,
                    f"{per_ch_mean[ch][idx]:.6g}",
                    f"{per_ch_pct95[ch][idx]:.6g}",
                    f"{vol_per_cell[idx]:.0f}",
                ])
        print(f"Wrote tracked-cell detail: {per_cell_csv}")

    return 0


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, required=True, help="Sample directory (contains <variant>.tif and <variant>_combined.tif)")
    ap.add_argument("--variant", choices=VALID_VARIANTS, default=DEFAULT_VARIANT)
    ap.add_argument("--track-cell", type=int, default=25, help="Cell id to track in the report (default: 25)")
    ap.add_argument("--no-clip-negatives", dest="clip_negatives", action="store_false", default=True)
    ap.add_argument("--bg-dilate-iters", type=int, default=2)
    ap.add_argument("--bg-top-clip-pct", type=float, default=1.0)
    ap.add_argument("--report-csv", type=Path, default=None)
    args = ap.parse_args(argv)
    return run(
        sample_dir=args.output_dir.resolve(),
        variant=args.variant,
        track_cell=args.track_cell,
        clip_negatives=args.clip_negatives,
        bg_dilate_iters=args.bg_dilate_iters,
        bg_top_clip_pct=args.bg_top_clip_pct,
        report_csv=args.report_csv,
    )


if __name__ == "__main__":
    sys.exit(main())
