#!/usr/bin/env python3
r"""
Background-statistics threshold exploration for per-channel intensity QC.

Loads each sample's mask + combined TIFF **once**, then evaluates many
thresholding strategies in memory (suitable for a single LSF job on large
volumes).

**Mean-threshold methods** (per channel ``T``, pass if ``mean_cell >= T`` unless
noted otherwise):

- ``bg_sigma``           — ``T = bg_mean + X * bg_std``
- ``bg_mad``             — ``T = bg_median + X * 1.4826 * MAD``
- ``otsu_or_bg_cells``   — ``T = max(log-Otsu on **per-cell means**, bg_sigma)``
  (legacy explore hybrid; distinct from production voxel hybrid)
- ``otsu2_or_bg_cells``  — ``T = max(Otsu2 on per-cell means, bg_sigma)``
- ``otsu_or_bg_voxel``   — ``T = max(log-Otsu on **positive voxels under mask**,
  bg_sigma)`` — matches ``qc/filter_by_intensity.py --method otsu_or_bg``
- ``otsu2_or_bg_voxel``  — ``T = max(otsu2 on voxel positives, bg_sigma)``
- ``bg_pct95``           — ``pct95_cell >= bg_sigma threshold``
- ``bg_frac``            — fraction of voxels ``>= T`` ≥ ``min_frac``

**Baselines** (``X`` = NaN in CSV):

- ``log_otsu`` / ``otsu2`` — thresholds from distributions of **per-cell means**
- ``log_otsu_voxel`` / ``otsu2_voxel`` — thresholds from **voxel** pools (mask>0)

**Optional 488 guardrails** (only when flags are set):

- ``bg_sigma_floor488`` — ``T_488 = max(bg_sigma_488, floor)``; sweep via
  ``--extra-floor-488``
- ``bg_sigma_pctfloor488`` — ``T_88 = max(bg_sigma_88, percentile(mean_488, p))``;
  sweep via ``--extra-mean-pct-floor-488``

**CLI additions**

- ``--gate-channels`` — comma list (default all three); adds ``n_pass_gate`` /
  ``track_cell_pass_gate``. ``n_pass_488560`` / ``track_cell_pass_488560`` are
  always AND(488,560) for comparison to production 488+560 gating.
- ``--qc-csv`` — merge ``pass_shape`` from an existing ``qc_features_filtered.csv``
  (same ``cell_id``); adds ``*_shape`` pass counts (intensity AND shape).

**Negative voxels:** by default voxels are clipped to ``>= 0`` before stats
(see ``--no-clip-negatives`` to match production ``filter_by_intensity`` bg
stats, which do not clip).

**HPC (one job per sample)**

Example ``bsub`` (run from repo root; use your real paths)::

   bsub -n 1 -q standard -J bgex -W 120 -M 32000 -R 'rusage[mem=32000]' \\
     -o logs/bg_explore.o -e logs/bg_explore.e \\
     bash -lc 'python /path/to/cmap/qc/bg_threshold_explore.py \\
       --output-dir /path/to/cmap/output/4_18_25/CGNSample3_Position6_decon_dsr \\
       --variant union_488_560 --track-cell 104 --no-clip-negatives \\
       --gate-channels 488,560 \\
       --qc-csv /path/to/cmap/output/4_18_25/.../cell_qc_union_488_560/qc_features_filtered.csv'

Output:

- ``<sample>/bg_threshold_report_<variant>.csv`` (or ``--report-csv`` path)
- ``<sample>/bg_threshold_report_<variant>_track_cell_detail.csv`` when the
  tracked label exists in the mask
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
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


def _load_filter_by_intensity_module():
    """Load sibling ``filter_by_intensity`` (``qc`` is not a Python package)."""
    fi_path = Path(__file__).resolve().parent / "filter_by_intensity.py"
    mod_name = "cmap_filter_by_intensity"
    spec = importlib.util.spec_from_file_location(mod_name, fi_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load spec for {fi_path}")
    mod = importlib.util.module_from_spec(spec)
    # Required so @dataclass and other deferred annotations resolve cls.__module__.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_gate_channels(spec: str | None) -> list[str]:
    if spec is None or str(spec).strip().lower() in ("", "all"):
        return list(CHANNEL_NAMES)
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    invalid = [p for p in parts if p not in CHANNEL_NAMES]
    if invalid:
        raise SystemExit(f"--gate-channels: unknown {invalid}; valid: {CHANNEL_NAMES}")
    return parts


def _load_pass_shape_by_cell_id(qc_csv: Path) -> dict[int, int]:
    by_id: dict[int, int] = {}
    with open(qc_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "cell_id" not in reader.fieldnames:
            raise SystemExit(f"{qc_csv}: missing cell_id column")
        if "pass_shape" not in reader.fieldnames:
            raise SystemExit(f"{qc_csv}: missing pass_shape column")
        for row in reader:
            cid = int(float(row["cell_id"]))
            by_id[cid] = int(float(row["pass_shape"]))
    return by_id


def _pass_shape_array(label_ids: np.ndarray, by_id: dict[int, int] | None) -> np.ndarray:
    """Boolean array aligned with ``label_ids``; default True when qc absent."""
    if not by_id:
        return np.ones(label_ids.size, dtype=bool)
    arr = np.ones(label_ids.size, dtype=bool)
    for i, lid in enumerate(label_ids):
        lid_i = int(lid)
        if lid_i in by_id:
            arr[i] = bool(by_id[lid_i])
    return arr


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
        slc = slices[lid - 1] if 0 < int(lid) <= max_lab else None
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
    per_ch_frac_above: dict[str, dict[float, np.ndarray]],
    bg: dict[str, dict[str, float]],
    otsu_log_T_cells: dict[str, float],
    otsu2_T_cells: dict[str, float],
    voxel_log_T: dict[str, float],
    voxel_otsu2_T: dict[str, float],
    label_ids: np.ndarray,
    track_cell: int | None,
    gate_channels: list[str],
    pass_shape_arr: np.ndarray,
    extra_floor_488_list: list[float],
    extra_pct_floor_488_list: list[int],
) -> list[dict[str, float | int | str]]:
    """Build result rows: one per (method, X, min_frac) plus optional sweeps."""
    n_cells = label_ids.size
    cell_idx_for_track: int | None = None
    if track_cell is not None:
        match = np.where(label_ids == track_cell)[0]
        if match.size:
            cell_idx_for_track = int(match[0])

    rows: list[dict[str, float | int | str]] = []
    gate_set = list(gate_channels)
    gate_str = ",".join(gate_channels)

    def _emit(
        method: str,
        X: float,
        min_frac: float,
        T: dict[str, float],
        pass_per_ch: dict[str, np.ndarray],
        *,
        extra_floor_488: str = "",
        extra_pct_floor_488: str = "",
    ) -> None:
        all_pass = np.ones(n_cells, dtype=bool)
        for ch in CHANNEL_NAMES:
            all_pass &= pass_per_ch[ch]
        n_pass = int(all_pass.sum())

        gate_pass = np.ones(n_cells, dtype=bool)
        for ch in gate_set:
            gate_pass &= pass_per_ch[ch]
        n_pass_gate = int(gate_pass.sum())

        pass_488560 = pass_per_ch["488"] & pass_per_ch["560"]
        n_pass_488560 = int(pass_488560.sum())

        all_pass_shape = all_pass & pass_shape_arr
        gate_pass_shape = gate_pass & pass_shape_arr
        pass_488560_shape = pass_488560 & pass_shape_arr

        row: dict[str, float | int | str] = {
            "method": method,
            "X": float(X),
            "min_frac": float(min_frac),
            "n_cells": int(n_cells),
            "n_pass_all3": n_pass,
            "gate_channels": gate_str,
            "n_pass_gate": n_pass_gate,
            # Human-readable: cells passing the gate_channels AND (same as n_pass_gate).
            "n_surviving_cells": n_pass_gate,
            "n_pass_488560": n_pass_488560,
            "n_pass_all3_shape": int(all_pass_shape.sum()),
            "n_pass_gate_shape": int(gate_pass_shape.sum()),
            "n_pass_488560_shape": int(pass_488560_shape.sum()),
            "extra_floor_488": extra_floor_488,
            "extra_pct_floor_488": extra_pct_floor_488,
        }
        if cell_idx_for_track is not None:
            row["track_cell_id"] = int(track_cell)  # type: ignore[arg-type]
            row["track_cell_pass_all3"] = int(bool(all_pass[cell_idx_for_track]))
            row["track_cell_pass_gate"] = int(bool(gate_pass[cell_idx_for_track]))
            # 1 = tracked cell is removed by this gate (fails), 0 = survives.
            row["track_cell_killed"] = int(not bool(gate_pass[cell_idx_for_track]))
            row["n_surviving_cells_shape"] = int(gate_pass_shape.sum())
            row["track_cell_killed_shape"] = int(not bool(gate_pass_shape[cell_idx_for_track]))
            row["track_cell_pass_488560"] = int(bool(pass_488560[cell_idx_for_track]))
            row["track_cell_killed_488560"] = int(not bool(pass_488560[cell_idx_for_track]))
            row["track_cell_pass_all3_shape"] = int(bool(all_pass_shape[cell_idx_for_track]))
            row["track_cell_pass_gate_shape"] = int(bool(gate_pass_shape[cell_idx_for_track]))
            row["track_cell_pass_488560_shape"] = int(bool(pass_488560_shape[cell_idx_for_track]))
            for ch in CHANNEL_NAMES:
                row[f"track_cell_pass_{ch}"] = int(bool(pass_per_ch[ch][cell_idx_for_track]))
        else:
            row["track_cell_id"] = int(track_cell) if track_cell is not None else ""
            row["track_cell_pass_all3"] = ""
            row["track_cell_pass_gate"] = ""
            row["track_cell_killed"] = ""
            row["n_surviving_cells_shape"] = int(gate_pass_shape.sum())
            row["track_cell_killed_shape"] = ""
            row["track_cell_pass_488560"] = ""
            row["track_cell_killed_488560"] = ""
            row["track_cell_pass_all3_shape"] = ""
            row["track_cell_pass_gate_shape"] = ""
            row["track_cell_pass_488560_shape"] = ""
            for ch in CHANNEL_NAMES:
                row[f"track_cell_pass_{ch}"] = ""
        for ch in CHANNEL_NAMES:
            row[f"T_{ch}"] = float(T[ch])
            row[f"n_pass_{ch}"] = int(pass_per_ch[ch].sum())
        rows.append(row)

    for X in X_SWEEP:
        T_sigma = {ch: bg[ch]["bg_mean"] + X * bg[ch]["bg_std"] for ch in CHANNEL_NAMES}
        T_mad = {ch: bg[ch]["bg_median"] + X * 1.4826 * bg[ch]["bg_mad"] for ch in CHANNEL_NAMES}
        T_hybrid_cells_log = {ch: max(otsu_log_T_cells[ch], T_sigma[ch]) for ch in CHANNEL_NAMES}
        T_hybrid_cells_otsu2 = {ch: max(otsu2_T_cells[ch], T_sigma[ch]) for ch in CHANNEL_NAMES}
        T_hybrid_voxel_log = {ch: max(voxel_log_T[ch], T_sigma[ch]) for ch in CHANNEL_NAMES}
        T_hybrid_voxel_otsu2 = {ch: max(voxel_otsu2_T[ch], T_sigma[ch]) for ch in CHANNEL_NAMES}

        pass_per_ch = {ch: per_ch_mean[ch] >= T_sigma[ch] for ch in CHANNEL_NAMES}
        _emit("bg_sigma", X, 0.0, T_sigma, pass_per_ch)

        pass_per_ch = {ch: per_ch_mean[ch] >= T_mad[ch] for ch in CHANNEL_NAMES}
        _emit("bg_mad", X, 0.0, T_mad, pass_per_ch)

        pass_per_ch = {ch: per_ch_mean[ch] >= T_hybrid_cells_log[ch] for ch in CHANNEL_NAMES}
        _emit("otsu_or_bg_cells", X, 0.0, T_hybrid_cells_log, pass_per_ch)

        pass_per_ch = {ch: per_ch_mean[ch] >= T_hybrid_cells_otsu2[ch] for ch in CHANNEL_NAMES}
        _emit("otsu2_or_bg_cells", X, 0.0, T_hybrid_cells_otsu2, pass_per_ch)

        pass_per_ch = {ch: per_ch_mean[ch] >= T_hybrid_voxel_log[ch] for ch in CHANNEL_NAMES}
        _emit("otsu_or_bg_voxel", X, 0.0, T_hybrid_voxel_log, pass_per_ch)

        pass_per_ch = {ch: per_ch_mean[ch] >= T_hybrid_voxel_otsu2[ch] for ch in CHANNEL_NAMES}
        _emit("otsu2_or_bg_voxel", X, 0.0, T_hybrid_voxel_otsu2, pass_per_ch)

        pass_per_ch = {ch: per_ch_pct95[ch] >= T_sigma[ch] for ch in CHANNEL_NAMES}
        _emit("bg_pct95", X, 0.0, T_sigma, pass_per_ch)

        for min_frac in MIN_FRAC_SWEEP:
            pass_per_ch = {ch: per_ch_frac_above[ch][X] >= min_frac for ch in CHANNEL_NAMES}
            _emit("bg_frac", X, min_frac, T_sigma, pass_per_ch)

        for floor in extra_floor_488_list:
            T_f = dict(T_sigma)
            T_f["488"] = max(T_sigma["488"], float(floor))
            pass_per_ch = {ch: per_ch_mean[ch] >= T_f[ch] for ch in CHANNEL_NAMES}
            _emit("bg_sigma_floor488", X, 0.0, T_f, pass_per_ch, extra_floor_488=str(floor))

        for pctl in extra_pct_floor_488_list:
            fl = float(np.percentile(per_ch_mean["488"], float(pctl)))
            T_p = dict(T_sigma)
            T_p["488"] = max(T_sigma["488"], fl)
            pass_per_ch = {ch: per_ch_mean[ch] >= T_p[ch] for ch in CHANNEL_NAMES}
            _emit(
                "bg_sigma_pctfloor488",
                X,
                0.0,
                T_p,
                pass_per_ch,
                extra_pct_floor_488=str(pctl),
            )

    pass_per_ch = {ch: per_ch_mean[ch] >= otsu_log_T_cells[ch] for ch in CHANNEL_NAMES}
    _emit("log_otsu", float("nan"), 0.0, otsu_log_T_cells, pass_per_ch)

    pass_per_ch = {ch: per_ch_mean[ch] >= otsu2_T_cells[ch] for ch in CHANNEL_NAMES}
    _emit("otsu2", float("nan"), 0.0, otsu2_T_cells, pass_per_ch)

    pass_per_ch = {ch: per_ch_mean[ch] >= voxel_log_T[ch] for ch in CHANNEL_NAMES}
    _emit("log_otsu_voxel", float("nan"), 0.0, voxel_log_T, pass_per_ch)

    pass_per_ch = {ch: per_ch_mean[ch] >= voxel_otsu2_T[ch] for ch in CHANNEL_NAMES}
    _emit("otsu2_voxel", float("nan"), 0.0, voxel_otsu2_T, pass_per_ch)

    return rows


def run(
    sample_dir: Path,
    variant: str,
    track_cell: int | None,
    clip_negatives: bool,
    bg_dilate_iters: int,
    bg_top_clip_pct: float,
    report_csv: Path | None,
    gate_channels: list[str],
    qc_csv: Path | None,
    extra_floor_488: list[float],
    extra_pct_floor_488: list[int],
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
    print(f"gate_channels={gate_channels}")
    if qc_csv is not None:
        print(f"qc_csv (pass_shape merge): {qc_csv}")

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
    otsu_log_T_cells: dict[str, float] = {}
    otsu2_T_cells: dict[str, float] = {}

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
        otsu_log_T_cells[ch] = _otsu_log(means)
        otsu2_T_cells[ch] = _otsu2(means)

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
            f"otsu_log_cells={otsu_log_T_cells[ch]:.4g}  otsu2_cells={otsu2_T_cells[ch]:.4g}"
        )

    _fbi = _load_filter_by_intensity_module()
    voxel_log_T = _fbi._compute_pixel_thresholds_from_volume(combined, mask_i, "log_otsu")
    voxel_otsu2_T = _fbi._compute_pixel_thresholds_from_volume(combined, mask_i, "otsu2")
    print(
        "  voxel thresholds (mask foreground, matches filter_by_intensity):  "
        + "  ".join(f"{ch} log={voxel_log_T[ch]:.4g} otsu2={voxel_otsu2_T[ch]:.4g}" for ch in CHANNEL_NAMES)
    )

    shape_by_id: dict[int, int] | None = None
    if qc_csv is not None:
        if not qc_csv.is_file():
            print(f"ERROR: --qc-csv not found: {qc_csv}")
            return 1
        shape_by_id = _load_pass_shape_by_cell_id(qc_csv.resolve())
    pass_shape_arr = _pass_shape_array(label_ids, shape_by_id)

    rows = _evaluate_methods(
        per_ch_mean=per_ch_mean,
        per_ch_pct95=per_ch_pct95,
        per_ch_frac_above=per_ch_frac_above,
        bg=bg,
        otsu_log_T_cells=otsu_log_T_cells,
        otsu2_T_cells=otsu2_T_cells,
        voxel_log_T=voxel_log_T,
        voxel_otsu2_T=voxel_otsu2_T,
        label_ids=label_ids,
        track_cell=track_cell,
        gate_channels=gate_channels,
        pass_shape_arr=pass_shape_arr,
        extra_floor_488_list=extra_floor_488,
        extra_pct_floor_488_list=extra_pct_floor_488,
    )

    print()
    print(f"=== Method comparison (track_cell={track_cell}, present={track_present}) ===")
    hdr = (
        f"{'method':<22} {'X':>5} {'min_frac':>9} "
        f"{'T_642':>10} {'T_488':>10} {'T_560':>10} "
        f"{'n_surv':>7} {'kill':>5} {'n488560':>8} "
        f"{'kill488560':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        x_val = r["X"]
        x_str = f"{x_val:.0f}" if not (isinstance(x_val, float) and np.isnan(x_val)) else "  -"
        n_sv = r.get("n_surviving_cells", r.get("n_pass_gate", "-"))
        k = r.get("track_cell_killed", "-")
        k56 = r.get("track_cell_killed_488560", "-")
        print(
            f"{str(r['method']):<22} {x_str:>5} {r['min_frac']:>9.2f} "
            f"{r['T_642']:>10.4g} {r['T_488']:>10.4g} {r['T_560']:>10.4g} "
            f"{str(n_sv):>7} {str(k):>5} {r['n_pass_488560']:>8d} "
            f"{str(k56):>11}"
        )

    if report_csv is None:
        report_csv = sample_dir / f"bg_threshold_report_{variant}.csv"

    fieldnames: list[str] = [
        "method",
        "X",
        "min_frac",
        "n_cells",
        "n_pass_all3",
        "track_cell_id",
        "track_cell_pass_all3",
        *[f"track_cell_pass_{ch}" for ch in CHANNEL_NAMES],
        "gate_channels",
        "n_pass_gate",
        "n_surviving_cells",
        "track_cell_pass_gate",
        "track_cell_killed",
        "n_pass_488560",
        "track_cell_pass_488560",
        "track_cell_killed_488560",
        "n_pass_all3_shape",
        "track_cell_pass_all3_shape",
        "n_pass_gate_shape",
        "n_surviving_cells_shape",
        "track_cell_pass_gate_shape",
        "track_cell_killed_shape",
        "n_pass_488560_shape",
        "track_cell_pass_488560_shape",
        "extra_floor_488",
        "extra_pct_floor_488",
    ]
    for ch in CHANNEL_NAMES:
        fieldnames.extend([f"T_{ch}", f"n_pass_{ch}"])
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
                w.writerow(
                    [
                        ch,
                        f"{per_ch_mean[ch][idx]:.6g}",
                        f"{per_ch_pct95[ch][idx]:.6g}",
                        f"{vol_per_cell[idx]:.0f}",
                    ]
                )
        print(f"Wrote tracked-cell detail: {per_cell_csv}")

    return 0


def _parse_float_list(s: str | None) -> list[float]:
    if not s or not str(s).strip():
        return []
    out: list[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def _parse_int_list(s: str | None) -> list[int]:
    if not s or not str(s).strip():
        return []
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, required=True, help="Sample directory (contains <variant>.tif and <variant>_combined.tif)")
    ap.add_argument("--variant", choices=VALID_VARIANTS, default=DEFAULT_VARIANT)
    ap.add_argument("--track-cell", type=int, default=25, help="Cell id to track in the report (default: 25)")
    ap.add_argument("--no-clip-negatives", dest="clip_negatives", action="store_false", default=True)
    ap.add_argument("--bg-dilate-iters", type=int, default=2)
    ap.add_argument("--bg-top-clip-pct", type=float, default=1.0)
    ap.add_argument("--report-csv", type=Path, default=None)
    ap.add_argument(
        "--gate-channels",
        type=str,
        default="all",
        help="Comma-separated channels for n_pass_gate / track_cell_pass_gate (default: all three).",
    )
    ap.add_argument(
        "--qc-csv",
        type=Path,
        default=None,
        help="Existing qc_features_filtered.csv with cell_id,pass_shape — adds *_shape pass counts.",
    )
    ap.add_argument(
        "--extra-floor-488",
        type=str,
        default="",
        help="Comma-separated absolute floors for 488 only (method bg_sigma_floor488), each combined with every X in the sweep.",
    )
    ap.add_argument(
        "--extra-mean-pct-floor-488",
        type=str,
        default="",
        help="Comma-separated percentiles p in [0,100] — T_488 = max(bg_sigma_488, percentile(mean_488, p)); method bg_sigma_pctfloor488.",
    )
    args = ap.parse_args(argv)
    gate_channels = _parse_gate_channels(args.gate_channels)
    extra_f = _parse_float_list(args.extra_floor_488)
    extra_p = _parse_int_list(args.extra_mean_pct_floor_488)
    return run(
        sample_dir=args.output_dir.resolve(),
        variant=args.variant,
        track_cell=args.track_cell,
        clip_negatives=args.clip_negatives,
        bg_dilate_iters=args.bg_dilate_iters,
        bg_top_clip_pct=args.bg_top_clip_pct,
        report_csv=args.report_csv,
        gate_channels=gate_channels,
        qc_csv=args.qc_csv.resolve() if args.qc_csv else None,
        extra_floor_488=extra_f,
        extra_pct_floor_488=extra_p,
    )


if __name__ == "__main__":
    sys.exit(main())
