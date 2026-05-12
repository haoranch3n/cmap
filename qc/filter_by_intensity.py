#!/usr/bin/env python3
"""
Filter cells by per-channel mean intensity and append pass/fail to QC CSV.

Supported ``--method`` values:
- ``log_otsu``         -- legacy default; Otsu over log-transformed positives.
- ``otsu``             -- Otsu over raw values.
- ``otsu2``            -- Otsu over values clipped to [0, p99.99].
- ``bg_sigma:X``       -- ``T = bg_mean + X * bg_std`` per channel, with bg
  estimated on the dilated-mask complement (un-clipped voxels, top P%
  trimmed). Recommended primary method (X=3 on Sample7_Position7 cleanly
  drops noise-floor cells like union_488_560 cell 25).
- ``otsu_or_bg:X``     -- hybrid floor: ``T = max(log_otsu_T, bg_mean+X*bg_std)``.
  Falls back to bg_sigma when log_otsu collapses; tightens to log_otsu when
  it actually finds bimodality.
- ``otsu_or_bg_voxel:X`` -- **same hybrid as** ``otsu_or_bg:X`` (independent
  ``T`` per channel from voxel log-Otsu + bg stats). **Requires** ``--from-volume``.
  Writes ``otsu_or_bg_voxel_threshold_{642,488,560}``,
  ``otsu_or_bg_voxel_pass_{642,488,560}``, and ``pass_otsu_or_bg_voxel_<chans>_shape``
  (no generic ``threshold_*`` columns). Use ``--cell-qc-dir-name`` to avoid
  overwriting the default union QC folder.
- ``--voxel-488-log-otsu-positive-pctile-lo P`` with ``--from-volume`` and
  ``otsu_or_bg*``: trim channel-488 positive voxels below the ``P``-th
  percentile of positives before voxel log-Otsu (``0`` = off; try ``1``–``2``).
- ``percentile:N`` / ``fixed:V`` -- legacy diagnostic options.

With ``--from-volume``, optional 3D shape gates (bbox aspect, fill ratio,
inertia anisotropy, two-step erosion split count) run first; labels that fail
receive ``pass_shape=0`` and ``pass_<method>_shape`` stays 0 even when the
intensity gate would pass (so ``apply_qc_pass_to_label_mask.py`` drops them).
The ``pass_<method>_shape`` column is named after the chosen method
(``pass_otsu_shape`` / ``pass_otsu2_shape`` / ``pass_bg_sigma_shape`` /
``pass_otsu_or_bg_shape`` / ``pass_otsu_or_bg_voxel_488560_shape``, etc.).
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from segmentation.config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    from config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT

from postprocess import DEFAULT_VARIANT, VALID_VARIANTS, variant_files

CHANNEL_NAMES = ["642", "488", "560"]
CHANNEL_INDICES = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}


@dataclass(frozen=True)
class ShapeFilterParams:
    """3D morphology gate (before Otsu pixel gates are combined into pass_qc)."""

    enabled: bool
    max_bbox_aspect: float
    min_fill_ratio: float
    max_inertia_ratio: float
    min_vol_erode_test: int
    min_vol_skip_all: int
    erode_iterations: int


def _shape_metrics_3d(bin3d: np.ndarray, erode_iterations: int = 2) -> dict[str, float]:
    """Cheap 3D shape descriptors on a boolean bbox crop (single label)."""
    v = float(int(bin3d.sum()))
    dz, dy, dx = bin3d.shape
    bbox_vol = float(max(dz * dy * dx, 1))
    edges = np.array([dz, dy, dx], dtype=np.float64)
    smin = max(float(edges.min()), 1.0)
    smax = float(edges.max())
    bbox_aspect = smax / smin
    fill_ratio = v / bbox_vol

    coords = np.argwhere(bin3d)
    if coords.shape[0] < 4:
        inertia_ratio = 1.0
    else:
        c = coords.astype(np.float64)
        c -= c.mean(axis=0, keepdims=True)
        cov = (c.T @ c) / float(c.shape[0])
        ev = np.linalg.eigvalsh(cov)
        ev = np.sort(ev)[::-1]
        inertia_ratio = float(ev[0] / max(ev[2], 1e-12))

    struct = ndimage.generate_binary_structure(3, 1)
    er = bin3d.astype(bool, copy=False)
    for _ in range(max(erode_iterations, 0)):
        er = ndimage.binary_erosion(er, structure=struct)
    if not er.any():
        erode_n_cc = 0.0
    else:
        _, erode_n_cc = ndimage.label(er, structure=struct)
        erode_n_cc = float(erode_n_cc)

    return {
        "shape_volume": v,
        "shape_bbox_aspect": float(bbox_aspect),
        "shape_fill_ratio": float(fill_ratio),
        "shape_inertia_ratio": float(inertia_ratio),
        "shape_erode2_n_cc": erode_n_cc,
    }


def _pass_shape(m: dict[str, float], p: ShapeFilterParams) -> bool:
    v = int(m["shape_volume"])
    if not p.enabled or v < p.min_vol_skip_all:
        return True

    if m["shape_bbox_aspect"] > p.max_bbox_aspect:
        return False
    if m["shape_fill_ratio"] < p.min_fill_ratio:
        return False
    if m["shape_inertia_ratio"] > p.max_inertia_ratio:
        return False

    if v >= p.min_vol_erode_test:
        ncc = int(round(m["shape_erode2_n_cc"]))
        if ncc >= 2:
            return False
        if ncc == 0:
            return False

    return True


DEFAULT_SHAPE_PARAMS = ShapeFilterParams(
    enabled=True,
    max_bbox_aspect=16.0,
    min_fill_ratio=0.045,
    max_inertia_ratio=20.0,
    min_vol_erode_test=400,
    min_vol_skip_all=50,
    erode_iterations=2,
)


def _resolve_dirs(args) -> tuple[Path, Path]:
    if args.data_rel:
        data_dir = Path(DATA_DIR) / args.data_rel
        output_dir = (
            Path(OUTPUT_DIR) / args.data_rel
            if OUTPUT_DIR != PROJECT_ROOT / "output"
            else PROJECT_ROOT / "output" / args.data_rel
        )
    elif args.output_dir:
        # Use absolute() not resolve() — resolve() follows all symlinks and can
        # convert /research/dept/... to an inaccessible /research_jude/... canonical
        # path on compute nodes where the mount alias differs.
        data_dir = args.output_dir.absolute()
        output_dir = args.output_dir.absolute()
    else:
        raise SystemExit("Provide either --data-rel or --output-dir")
    return data_dir, output_dir


def _parse_method(method_str: str) -> tuple[str, float | None]:
    if method_str == "log_otsu":
        return "log_otsu", None
    if method_str == "otsu":
        return "otsu", None
    if method_str == "otsu2":
        return "otsu2", None
    if method_str.startswith("bg_sigma:"):
        val = float(method_str.split(":", 1)[1])
        if val <= 0:
            raise ValueError(f"bg_sigma X must be > 0, got {val}")
        return "bg_sigma", val
    if method_str.startswith("otsu_or_bg:"):
        val = float(method_str.split(":", 1)[1])
        if val <= 0:
            raise ValueError(f"otsu_or_bg X must be > 0, got {val}")
        return "otsu_or_bg", val
    if method_str.startswith("otsu_or_bg_voxel:"):
        val = float(method_str.split(":", 1)[1])
        if val <= 0:
            raise ValueError(f"otsu_or_bg_voxel X must be > 0, got {val}")
        return "otsu_or_bg_voxel", val
    if method_str.startswith("percentile:"):
        val = float(method_str.split(":", 1)[1])
        if not 0 < val < 100:
            raise ValueError(f"Percentile must be in (0, 100), got {val}")
        return "percentile", val
    if method_str.startswith("fixed:"):
        return "fixed", float(method_str.split(":", 1)[1])
    raise ValueError(
        "Unknown method. Use 'log_otsu', 'otsu', 'otsu2', "
        "'bg_sigma:X', 'otsu_or_bg:X', 'otsu_or_bg_voxel:X', "
        "'percentile:N', or 'fixed:V'."
    )


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
    if method == "otsu2":
        clipped = np.clip(values, 0, np.percentile(values, 99.99))
        try:
            return float(threshold_otsu(clipped))
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


def _compute_pixel_thresholds_from_volume(
    combined_zcyx: np.ndarray,
    mask: np.ndarray,
    method: str = "log_otsu",
    voxel_488_positive_pctile_lo: float = 0.0,
) -> dict[str, float]:
    """Apply thresholding method on positive voxels under label foreground.

    ``voxel_488_positive_pctile_lo`` (Method A): for channel 488 only, after
    ``pos = vals[vals > 0]``, optionally drop voxels below the
    ``p``-th percentile of ``pos`` (``p`` in ``[0, 50)``) before ``log``+Otsu,
    to reduce denormal / near-zero dominance. ``0`` disables. Other channels
    unchanged.
    """
    from skimage.filters import threshold_otsu

    thresholds: dict[str, float] = {}
    fg = mask > 0
    p_lo = float(voxel_488_positive_pctile_lo)
    for ch_name, ch_idx in CHANNEL_INDICES.items():
        vals = combined_zcyx[:, ch_idx, :, :][fg].astype(np.float64).ravel()
        pos = vals[vals > 0]
        if len(pos) < 2:
            thresholds[ch_name] = 0.0
            continue
        if (
            method == "log_otsu"
            and ch_name == "488"
            and p_lo > 0.0
            and p_lo < 50.0
        ):
            cut = float(np.percentile(pos, p_lo))
            pos_trim = pos[pos >= cut]
            if pos_trim.size >= 2:
                pos = pos_trim
        try:
            if method == "log_otsu":
                thresholds[ch_name] = float(np.exp(float(threshold_otsu(np.log(pos)))))
            elif method == "otsu2":
                clipped = np.clip(pos, 0, np.percentile(pos, 99.99))
                thresholds[ch_name] = float(threshold_otsu(clipped))
            else:
                thresholds[ch_name] = float(np.exp(float(threshold_otsu(np.log(pos)))))
        except ValueError:
            thresholds[ch_name] = 0.0
    return thresholds


def _compute_background_stats(
    combined_zcyx: np.ndarray,
    mask: np.ndarray,
    dilate_iters: int = 2,
    top_clip_pct: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Per-channel background mean/std over the (dilated) cell-mask complement.

    Negative voxels are intentionally NOT clipped to 0: they carry information
    about the real noise scale (deconvolution overshoot occurs in both
    directions). Clipping the bg to >= 0 collapses bg_std and breaks any
    bg_mean + X * bg_std threshold. See ``.cursor/plans/filter_dev.md``.

    The top ``top_clip_pct`` percent of background voxels is dropped before
    computing mean/std as a debris guard (autofluorescent pixels that survive
    the dilated-complement). ``dilate_iters=2`` pulls the bg sample away from
    cell edges to avoid signal bleed.
    """
    fg = mask > 0
    if dilate_iters > 0:
        struct = ndimage.generate_binary_structure(3, 1)
        fg_dilated = ndimage.binary_dilation(fg, structure=struct, iterations=dilate_iters)
    else:
        fg_dilated = fg
    bg_mask = ~fg_dilated

    out: dict[str, dict[str, float]] = {}
    for ch_name, ch_idx in CHANNEL_INDICES.items():
        bg_vals = combined_zcyx[:, ch_idx, :, :][bg_mask].astype(np.float64, copy=False)
        if bg_vals.size == 0:
            out[ch_name] = {"bg_mean": 0.0, "bg_std": 0.0, "bg_n": 0}
            continue
        if top_clip_pct > 0:
            cutoff = float(np.percentile(bg_vals, 100.0 - top_clip_pct))
            bg_vals = bg_vals[bg_vals <= cutoff]
        if bg_vals.size == 0:
            out[ch_name] = {"bg_mean": 0.0, "bg_std": 0.0, "bg_n": 0}
            continue
        out[ch_name] = {
            "bg_mean": float(np.mean(bg_vals)),
            "bg_std": float(np.std(bg_vals)),
            "bg_n": int(bg_vals.size),
        }
    return out


def _log_otsu_per_channel(
    combined_zcyx: np.ndarray,
    mask: np.ndarray,
    voxel_488_positive_pctile_lo: float = 0.0,
) -> dict[str, float]:
    """Per-channel log_otsu over positive voxels under label foreground.

    Helper for the otsu_or_bg hybrid. Equivalent to
    ``_compute_pixel_thresholds_from_volume(method='log_otsu')`` but kept
    separate so the bg_sigma / otsu_or_bg paths can request both bg-stats and
    log_otsu values from one shared computation surface.
    """
    return _compute_pixel_thresholds_from_volume(
        combined_zcyx,
        mask,
        method="log_otsu",
        voxel_488_positive_pctile_lo=voxel_488_positive_pctile_lo,
    )


def _save_bg_stats(bg_stats: dict[str, dict[str, float]], qc_dir: Path) -> None:
    """Write bg_mean/bg_std/bg_n per channel to ``qc_dir/bg_stats.csv`` (one row)."""
    qc_dir.mkdir(parents=True, exist_ok=True)
    row: dict[str, str] = {}
    for ch in CHANNEL_NAMES:
        row[f"bg_mean_{ch}"] = str(round(bg_stats[ch]["bg_mean"], 6))
        row[f"bg_std_{ch}"] = str(round(bg_stats[ch]["bg_std"], 6))
        row[f"bg_n_{ch}"] = str(int(bg_stats[ch]["bg_n"]))
    fieldnames = list(row.keys())
    out_csv = qc_dir / "bg_stats.csv"
    tmp = str(out_csv) + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    os.replace(tmp, str(out_csv))
    print(f"  Wrote bg_stats: {out_csv}")


def _bg_sigma_thresholds(
    bg_stats: dict[str, dict[str, float]], x: float
) -> dict[str, float]:
    return {ch: bg_stats[ch]["bg_mean"] + x * bg_stats[ch]["bg_std"] for ch in CHANNEL_NAMES}


def _otsu_or_bg_thresholds(
    bg_stats: dict[str, dict[str, float]], log_otsu_T: dict[str, float], x: float
) -> dict[str, float]:
    bg_T = _bg_sigma_thresholds(bg_stats, x)
    return {ch: max(float(log_otsu_T[ch]), float(bg_T[ch])) for ch in CHANNEL_NAMES}


def _method_suffix(method: str, filter_channels: list[str] | None = None) -> str:
    """Column suffix used in ``pass_<suffix>_shape`` for the chosen method.

    When ``filter_channels`` is a strict subset of all channels the channel
    names are appended without separators, e.g. ``bg_sigma_488560``.
    """
    if method == "otsu2":
        base = "otsu2"
    elif method == "bg_sigma":
        base = "bg_sigma"
    elif method == "otsu_or_bg":
        base = "otsu_or_bg"
    elif method == "otsu_or_bg_voxel":
        base = "otsu_or_bg_voxel"
    else:
        base = "otsu"

    if filter_channels is not None and set(filter_channels) != set(CHANNEL_NAMES):
        # Sort by the canonical channel order so the suffix is deterministic.
        ordered = [ch for ch in CHANNEL_NAMES if ch in filter_channels]
        base = base + "_" + "".join(ordered)
    return base


def run_from_volume(
    output_dir: Path,
    method_str: str = "log_otsu",
    force: bool = False,
    variant: str = DEFAULT_VARIANT,
    shape_params: ShapeFilterParams | None = None,
    bg_dilate_iters: int = 2,
    bg_top_clip_pct: float = 1.0,
    filter_channels: list[str] | None = None,
    cell_qc_dir_name: str | None = None,
    voxel_488_positive_pctile_lo: float = 0.0,
) -> int:
    """Intensity QC from variant mask + combined TIFF (no cell crops required).

    ``filter_channels`` restricts which channels must pass the intensity gate.
    All channels are still measured; only the listed ones gate ``pass_intensity``
    and ``pass_<method>_shape``. Defaults to all channels when ``None``.
    """
    sp = shape_params or DEFAULT_SHAPE_PARAMS
    v = variant_files(variant)
    print(
        f"Variant: {variant}  --from-volume  (mask={v['mask']}, combined={v['combined']})"
    )
    if sp.enabled:
        print(
            "  shape_filter: on  "
            f"max_bbox_aspect={sp.max_bbox_aspect}  min_fill={sp.min_fill_ratio}  "
            f"max_inertia={sp.max_inertia_ratio}  min_vol_erode={sp.min_vol_erode_test}  "
            f"min_vol_skip={sp.min_vol_skip_all}  erode_iter={sp.erode_iterations}"
        )
    else:
        print("  shape_filter: off")
    qc_rel = cell_qc_dir_name if cell_qc_dir_name else v["cell_qc"]
    qc_dir = output_dir / qc_rel
    if cell_qc_dir_name:
        print(f"  cell_qc dir (override): {qc_rel}")
    if voxel_488_positive_pctile_lo > 0:
        print(
            f"  voxel_488_positive_pctile_lo={voxel_488_positive_pctile_lo} "
            "(trim 488 positives before voxel log-Otsu)"
        )
    output_csv = qc_dir / "qc_features_filtered.csv"
    if output_csv.exists() and not force:
        print(f"SKIP (exists): {output_csv}  (use --force to overwrite)")
        return 0

    mask_path = output_dir / v["mask"]
    combined_path = output_dir / v["combined"]
    if not mask_path.is_file():
        print(f"ERROR: mask not found: {mask_path}")
        return 1
    if not combined_path.is_file():
        print(f"ERROR: combined not found: {combined_path}")
        return 1

    mask = tifffile.imread(str(mask_path))
    if mask.ndim == 4:
        mask = mask[:, 0]
    if mask.ndim != 3:
        print(f"ERROR: expected 3-D mask, got {mask.shape}")
        return 1
    mask_i = mask.astype(np.int32, copy=False)

    combined = tifffile.imread(str(combined_path))
    if combined.ndim != 4 or combined.shape[1] < 3:
        print(f"ERROR: expected (Z, C>=3, Y, X) combined, got {combined.shape}")
        return 1
    z_min = min(mask_i.shape[0], combined.shape[0])
    mask_i = mask_i[:z_min]
    combined = combined[:z_min]

    label_ids = np.unique(mask_i)
    label_ids = label_ids[label_ids > 0]
    if label_ids.size == 0:
        print("ERROR: mask has no positive labels")
        return 1

    rows: list[dict[str, float | int]] = [
        {"cell_id": int(lid)} for lid in label_ids
    ]
    for ch_name, ch_idx in CHANNEL_INDICES.items():
        slab = combined[:, ch_idx, :, :].astype(np.float64, copy=False)
        means = ndimage.mean(slab, labels=mask_i, index=label_ids)
        for i in range(len(label_ids)):
            rows[i][f"mean_{ch_name}"] = float(means[i])

    method, param = _parse_method(method_str)

    if method in ("bg_sigma", "otsu_or_bg", "otsu_or_bg_voxel"):
        x_val = float(param) if param is not None else 3.0
        print(
            f"  bg_stats: dilate_iters={bg_dilate_iters}  top_clip_pct={bg_top_clip_pct}  "
            f"method={method}  X={x_val}"
        )
        bg_stats = _compute_background_stats(
            combined, mask_i, dilate_iters=bg_dilate_iters, top_clip_pct=bg_top_clip_pct
        )
        for ch in CHANNEL_NAMES:
            print(
                f"    ch {ch}: bg_mean={bg_stats[ch]['bg_mean']:.4g}  "
                f"bg_std={bg_stats[ch]['bg_std']:.4g}  bg_n={bg_stats[ch]['bg_n']:,d}"
            )
        _save_bg_stats(bg_stats, qc_dir)
        if method in ("otsu_or_bg", "otsu_or_bg_voxel"):
            log_otsu_T = _log_otsu_per_channel(
                combined, mask_i, voxel_488_positive_pctile_lo=voxel_488_positive_pctile_lo
            )
            thresholds = _otsu_or_bg_thresholds(bg_stats, log_otsu_T, x_val)
        else:
            thresholds = _bg_sigma_thresholds(bg_stats, x_val)
        # bg-anchored thresholds use the same value for cell-mean and pixel
        # paths (both are compared against the per-cell mean below); keep both
        # CSV columns populated so downstream consumers don't need to branch.
        px_thresholds = dict(thresholds)
    else:
        thresholds = {}
        for ch in CHANNEL_NAMES:
            values = np.array([float(r[f"mean_{ch}"]) for r in rows], dtype=np.float64)
            thresholds[ch] = _compute_threshold(values, method, param)
        px_thresholds = _compute_pixel_thresholds_from_volume(combined, mask_i, method=method)

    max_lab = int(mask_i.max())
    label_slices = ndimage.find_objects(mask_i, max_lab)

    str_rows: list[dict[str, str]] = []
    n_shape_fail = 0
    n_px_fail = 0

    # Channels that must pass to gate the combined pass flags.
    filter_set = set(filter_channels) if filter_channels else set(CHANNEL_NAMES)
    if filter_channels and set(filter_channels) != set(CHANNEL_NAMES):
        print(f"  filter_channels: {sorted(filter_channels)}  (642 not required to pass)")
    method_suffix = _method_suffix(method, filter_channels)

    for row in rows:
        cid = int(row["cell_id"])
        out: dict[str, str] = {"cell_id": str(cid)}
        if 0 < cid <= max_lab and label_slices[cid - 1] is not None:
            slc = label_slices[cid - 1]
            bin3d = mask_i[slc] == cid
        else:
            bin3d = mask_i == cid
        sm = _shape_metrics_3d(bin3d, erode_iterations=sp.erode_iterations)
        pass_shape = _pass_shape(sm, sp)
        out["shape_volume"] = str(int(sm["shape_volume"]))
        out["shape_bbox_aspect"] = str(round(sm["shape_bbox_aspect"], 4))
        out["shape_fill_ratio"] = str(round(sm["shape_fill_ratio"], 6))
        out["shape_inertia_ratio"] = str(round(sm["shape_inertia_ratio"], 4))
        out["shape_erode2_n_cc"] = str(int(round(sm["shape_erode2_n_cc"])))
        out["pass_shape"] = str(int(pass_shape))
        if not pass_shape:
            n_shape_fail += 1

        all_pass = True
        all_px_pass = True
        for ch in CHANNEL_NAMES:
            val = float(row[f"mean_{ch}"])
            passed = int(val >= thresholds[ch])
            out[f"mean_{ch}"] = str(round(val, 6))
            if method == "otsu_or_bg_voxel":
                out[f"otsu_or_bg_voxel_threshold_{ch}"] = str(round(thresholds[ch], 4))
                out[f"otsu_or_bg_voxel_pass_{ch}"] = str(passed)
            else:
                out[f"threshold_{ch}"] = str(round(thresholds[ch], 4))
                out[f"pass_{ch}"] = str(passed)
            if ch in filter_set and not passed:
                all_pass = False

            px_passed = int(val >= px_thresholds[ch])
            if method != "otsu_or_bg_voxel":
                out[f"px_threshold_{ch}"] = str(round(px_thresholds[ch], 4))
                out[f"px_pass_{ch}"] = str(px_passed)
            if ch in filter_set and not px_passed:
                all_px_pass = False
        out["pass_intensity"] = str(int(all_pass))
        px_only = int(all_px_pass)
        if px_only == 0:
            n_px_fail += 1
        out[f"pass_{method_suffix}_shape"] = str(int(px_only and pass_shape))
        str_rows.append(out)

    shape_cols = [
        "shape_volume",
        "shape_bbox_aspect",
        "shape_fill_ratio",
        "shape_inertia_ratio",
        "shape_erode2_n_cc",
        "pass_shape",
    ]
    if method == "otsu_or_bg_voxel":
        tail_cols: list[str] = []
        for ch in CHANNEL_NAMES:
            tail_cols.extend(
                [
                    f"otsu_or_bg_voxel_threshold_{ch}",
                    f"otsu_or_bg_voxel_pass_{ch}",
                ]
            )
        tail_cols.extend(["pass_intensity", f"pass_{method_suffix}_shape"])
    else:
        tail_cols = []
        for ch in CHANNEL_NAMES:
            tail_cols.extend(
                [f"threshold_{ch}", f"pass_{ch}", f"px_threshold_{ch}", f"px_pass_{ch}"]
            )
        tail_cols.extend(["pass_intensity", f"pass_{method_suffix}_shape"])
    base_cols = ["cell_id", "mean_642", "mean_488", "mean_560"]
    fieldnames = base_cols + shape_cols + tail_cols
    print(
        f"  shape: failed={n_shape_fail}  intensity_px_failed={n_px_fail}  "
        f"pass_mask={sum(int(r[f'pass_{method_suffix}_shape']) for r in str_rows)}/{len(str_rows)}"
    )

    qc_dir.mkdir(parents=True, exist_ok=True)
    tmp = str(output_csv) + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(str_rows)
    os.replace(tmp, str(output_csv))
    print(f"Wrote {output_csv}  ({len(str_rows)} cells, precrop pass columns only)")
    return 0


def run(
    output_dir: Path,
    method_str: str = "otsu",
    force: bool = False,
    variant: str = DEFAULT_VARIANT,
) -> int:
    v = variant_files(variant)
    print(f"Variant: {variant}  (cell_box={v['cell_box']}, cell_qc={v['cell_qc']})")
    qc_dir = output_dir / v["cell_qc"]
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

    box_dir = output_dir / v["cell_box"]
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
        row["pass_qc"] = int(all_px_pass)

    new_cols = []
    for ch in CHANNEL_NAMES:
        new_cols.extend([f"threshold_{ch}", f"pass_{ch}", f"px_threshold_{ch}", f"px_pass_{ch}"])
    new_cols.extend(["pass_intensity", "pass_qc"])
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
    ap.add_argument(
        "--from-volume",
        action="store_true",
        help=(
            "Compute per-cell means and Otsu pass columns from variant mask + combined "
            "TIFF only (no qc_features.csv or cell_boxing required). Writes precrop "
            "qc_features_filtered.csv for apply_qc_pass_to_label_mask.py."
        ),
    )
    ap.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default=DEFAULT_VARIANT,
        help=f"Mask variant whose cell_box/cell_qc dirs to use (default: {DEFAULT_VARIANT})",
    )
    ap.add_argument(
        "--shape-filter",
        choices=("on", "off"),
        default="on",
        help="With --from-volume: drop non-compact 3D labels before Otsu gates feed pass_qc (default: on).",
    )
    ap.add_argument("--shape-max-bbox-aspect", type=float, default=DEFAULT_SHAPE_PARAMS.max_bbox_aspect)
    ap.add_argument("--shape-min-fill", type=float, default=DEFAULT_SHAPE_PARAMS.min_fill_ratio)
    ap.add_argument("--shape-max-inertia-ratio", type=float, default=DEFAULT_SHAPE_PARAMS.max_inertia_ratio)
    ap.add_argument("--shape-min-vol-erode", type=int, default=DEFAULT_SHAPE_PARAMS.min_vol_erode_test)
    ap.add_argument("--shape-min-vol-skip", type=int, default=DEFAULT_SHAPE_PARAMS.min_vol_skip_all)
    ap.add_argument("--shape-erode-iterations", type=int, default=DEFAULT_SHAPE_PARAMS.erode_iterations)
    ap.add_argument(
        "--bg-dilate-iters",
        type=int,
        default=2,
        help="bg_sigma/otsu_or_bg: dilate the cell mask by N voxels before taking complement (default: 2).",
    )
    ap.add_argument(
        "--bg-top-clip-pct",
        type=float,
        default=1.0,
        help="bg_sigma/otsu_or_bg: drop top P%% of background voxels as debris before mean/std (default: 1.0).",
    )
    ap.add_argument(
        "--filter-channels",
        type=str,
        default=None,
        help=(
            "Comma-separated channel names that must pass the intensity gate "
            "(default: all channels). E.g. '488,560' to require only those two "
            "channels and treat 642/DAPI as non-gating. All channels are still "
            "measured; only the listed ones affect pass_intensity and "
            "pass_<method>_shape."
        ),
    )
    ap.add_argument(
        "--cell-qc-dir-name",
        type=str,
        default=None,
        help=(
            "With --from-volume: write QC CSV and bg_stats under "
            "OUTPUT_DIR/<this> instead of the variant's default cell_qc folder "
            "(relative name, e.g. cell_qc_union_488_560_otsu_or_bg_voxel)."
        ),
    )
    ap.add_argument(
        "--voxel-488-log-otsu-positive-pctile-lo",
        type=float,
        default=0.0,
        help=(
            "With --from-volume and otsu_or_bg / otsu_or_bg_voxel: for 488 only, "
            "drop positive voxels below this percentile of the positive-voxel pool "
            "before voxel log-Otsu (0 disables; typical sweep 0.5–3)."
        ),
    )
    args = ap.parse_args()
    _, output_dir = _resolve_dirs(args)
    if float(args.voxel_488_log_otsu_positive_pctile_lo) < 0 or float(
        args.voxel_488_log_otsu_positive_pctile_lo
    ) >= 50:
        raise SystemExit("--voxel-488-log-otsu-positive-pctile-lo must be in [0, 50)")
    if (
        float(args.voxel_488_log_otsu_positive_pctile_lo) > 0
        and args.from_volume
        and not str(args.method).startswith(("otsu_or_bg", "otsu_or_bg_voxel"))
    ):
        raise SystemExit(
            "--voxel-488-log-otsu-positive-pctile-lo only applies with "
            "--method otsu_or_bg:* or otsu_or_bg_voxel:*"
        )
    if not args.from_volume and str(args.method).startswith("otsu_or_bg_voxel"):
        raise SystemExit("otsu_or_bg_voxel:* requires --from-volume")
    if args.from_volume:
        sp = ShapeFilterParams(
            enabled=args.shape_filter == "on",
            max_bbox_aspect=args.shape_max_bbox_aspect,
            min_fill_ratio=args.shape_min_fill,
            max_inertia_ratio=args.shape_max_inertia_ratio,
            min_vol_erode_test=args.shape_min_vol_erode,
            min_vol_skip_all=args.shape_min_vol_skip,
            erode_iterations=max(0, int(args.shape_erode_iterations)),
        )
        fc: list[str] | None = None
        if args.filter_channels:
            fc = [c.strip() for c in args.filter_channels.split(",") if c.strip()]
            invalid = [c for c in fc if c not in CHANNEL_NAMES]
            if invalid:
                raise SystemExit(
                    f"--filter-channels: unknown channel(s) {invalid}. "
                    f"Valid: {list(CHANNEL_NAMES)}"
                )
        return run_from_volume(
            output_dir,
            method_str=args.method,
            force=args.force,
            variant=args.variant,
            shape_params=sp,
            bg_dilate_iters=max(0, int(args.bg_dilate_iters)),
            bg_top_clip_pct=max(0.0, float(args.bg_top_clip_pct)),
            filter_channels=fc,
            cell_qc_dir_name=args.cell_qc_dir_name,
            voxel_488_positive_pctile_lo=float(args.voxel_488_log_otsu_positive_pctile_lo),
        )
    return run(output_dir, method_str=args.method, force=args.force, variant=args.variant)


if __name__ == "__main__":
    sys.exit(main())

