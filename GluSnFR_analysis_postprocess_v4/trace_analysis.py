"""
Trace Analysis Module - Firing Detection, Quantification, and Visualization

Analyses synapse traces from combined CSV data:
  - Detects accepted firing events (local maxima above rolling threshold)
  - Quantifies evoked stimulus responses
  - Generates summary statistics at ROI, image, and group levels
  - Produces publication-ready plots

Based on: iGluSnFR_analysis_toolbox/synapse_analysis.py

Output directory layout (under <output>/csv_outputs/analysis/):
    csvs/
        combined_with_firings.csv      # combined CSV + accepted_peak column
        firings_per_roi.csv            # one row per ROI
        firings_per_image.csv          # one row per image
        group_summary.csv              # one row per group+condition
        analysis_params.json           # parameters used
    plots/
        spontaneous/
            traces_<GROUP>.png
            traces_all_groups.png
            firings_boxplot_by_group.png
            peak_intensity_histogram.png
            peak_intensity_boxplot_by_group.png
        evoked/
            <N>AP_<F>Hz/
                firings_boxplot_by_group.png
                response_rate_boxplot_by_group.png
                peak_intensity_histogram.png
                peak_intensity_boxplot_by_group.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from pathlib import Path

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


def _save_csv(df, path):
    """Save a pandas DataFrame to CSV, using polars for speed when available."""
    if _HAS_POLARS:
        try:
            pl.from_pandas(df).write_csv(path)
            return
        except Exception:
            pass  # fall through to pandas
    df.to_csv(path, index=False)


# =========================================================================
# Firing detection
# =========================================================================

def detect_firings(signal, frames, sd_multiplier=4.0, rolling_window=10,
                   baseline_start_frame=0, baseline_end_frame=49,
                   order=1,
                   # Deprecated – kept for backward-compat callers
                   baseline_frames=None):
    """
    Detect accepted firing events in a single ROI trace.

    A firing is a local maximum whose value exceeds::

        rolling_mean + sd_multiplier * baseline_SD

    Parameters
    ----------
    signal : array-like
        Intensity values.
    frames : array-like
        Corresponding frame numbers.
    sd_multiplier : float
        Multiplier for baseline SD to set threshold.
    rolling_window : int
        Window size for the centred rolling mean.
    baseline_start_frame, baseline_end_frame : int
        Frame numbers (inclusive) that define the baseline window for
        estimating SD.  Defaults to 0–49 (the pre-stimulus period for
        standard evoked protocols).  Both values are clamped to the
        actual trace range so short recordings never cause errors.
    order : int
        Points on each side for ``argrelextrema`` local-maximum comparison.
    baseline_frames : int or None
        **Deprecated.**  If provided (and the new start/end params are at
        their defaults), falls back to the old behaviour of using the last
        *N* frames.

    Returns
    -------
    dict
        accepted_frames, accepted_values, accepted_indices,
        all_local_max_frames, threshold, rolling_mean,
        baseline_sd, baseline_mean, baseline_median, baseline_mad,
        baseline_p5, baseline_p95, baseline_slope
    """
    signal = np.asarray(signal, dtype=float)
    frames = np.asarray(frames)
    n = len(signal)

    # --- Determine baseline slice (indices into *signal* array) ---
    if baseline_frames is not None and baseline_start_frame == 0 and baseline_end_frame == 49:
        # Legacy caller: use last N frames
        bl_idx_start = max(0, n - int(baseline_frames))
        bl_idx_end = n
    else:
        # New behaviour: map frame numbers to array indices
        frame_arr = frames if len(frames) == n else np.arange(n)
        bl_idx_start = int(np.searchsorted(frame_arr, baseline_start_frame, side="left"))
        bl_idx_end = int(np.searchsorted(frame_arr, baseline_end_frame, side="right"))
        # Clamp to valid range
        bl_idx_start = max(0, min(bl_idx_start, n - 1))
        bl_idx_end = max(bl_idx_start + 1, min(bl_idx_end, n))

    baseline = signal[bl_idx_start:bl_idx_end]
    if len(baseline) == 0:
        baseline = signal  # ultimate fallback
    baseline_sd = float(np.std(baseline))
    baseline_mean = float(np.mean(baseline))
    baseline_median = float(np.median(baseline))
    baseline_mad = float(np.median(np.abs(baseline - baseline_median)))
    baseline_p5 = float(np.percentile(baseline, 5))
    baseline_p95 = float(np.percentile(baseline, 95))
    # Baseline drift (linear regression slope over the baseline window)
    if len(baseline) >= 2:
        _x = np.arange(len(baseline), dtype=float)
        _x -= _x.mean()
        baseline_slope = float(np.dot(_x, baseline - baseline_mean) /
                               np.dot(_x, _x)) if np.dot(_x, _x) > 0 else 0.0
    else:
        baseline_slope = 0.0

    # Rolling mean
    s = pd.Series(signal)
    rolling_mean = s.rolling(
        window=rolling_window, center=True, min_periods=1
    ).mean().values

    # Threshold
    threshold = rolling_mean + sd_multiplier * baseline_sd

    empty = {
        "accepted_frames": np.array([]),
        "accepted_values": np.array([]),
        "accepted_indices": np.array([], dtype=int),
        "all_local_max_frames": np.array([]),
        "threshold": threshold,
        "rolling_mean": rolling_mean,
        "baseline_sd": baseline_sd,
        "baseline_mean": baseline_mean,
        "baseline_median": baseline_median,
        "baseline_mad": baseline_mad,
        "baseline_p5": baseline_p5,
        "baseline_p95": baseline_p95,
        "baseline_slope": baseline_slope,
    }

    if len(signal) < 3:
        return empty

    local_max_idx = argrelextrema(signal, np.greater, order=order)[0]
    if len(local_max_idx) == 0:
        return empty

    accepted_idx = np.array(
        [i for i in local_max_idx if signal[i] > threshold[i]]
    )

    return {
        "accepted_frames": frames[accepted_idx] if len(accepted_idx) else np.array([]),
        "accepted_values": signal[accepted_idx] if len(accepted_idx) else np.array([]),
        "accepted_indices": accepted_idx,
        "all_local_max_frames": frames[local_max_idx],
        "threshold": threshold,
        "rolling_mean": rolling_mean,
        "baseline_sd": baseline_sd,
        "baseline_mean": baseline_mean,
        "baseline_median": baseline_median,
        "baseline_mad": baseline_mad,
        "baseline_p5": baseline_p5,
        "baseline_p95": baseline_p95,
        "baseline_slope": baseline_slope,
    }


# =========================================================================
# Evoked stimulus handling
# =========================================================================

def compute_stim_frames(n_ap, stim_hz, frame_rate, stim_start=50):
    """
    Compute stimulus frame numbers for evoked experiments.

    stim_interval = round(frame_rate / stim_hz)  (frames between stimuli)
    Last stimulus at: stim_start + (n_ap - 1) * stim_interval

    Example: 100 AP at 20 Hz, frame_rate=100 Hz, stim_start=50
        interval = 5, last stim frame = 50 + 99*5 = 545
    """
    if stim_hz <= 0 or frame_rate <= 0:
        return [], 0
    stim_interval = round(frame_rate / stim_hz)
    if stim_interval < 1:
        stim_interval = 1
    stim_frames = [stim_start + i * stim_interval for i in range(int(n_ap))]
    return stim_frames, stim_interval


def quantify_evoked_responses(accepted_frames, stim_frames, response_window=5):
    """
    Count how many stimuli received a response within *response_window*
    frames after each stimulus.
    """
    firing_set = set(int(f) for f in accepted_frames)
    responses = 0
    responding_stims = []

    for sf in stim_frames:
        sf_int = int(sf)
        if any(f in firing_set
               for f in range(sf_int + 1, sf_int + response_window + 1)):
            responses += 1
            responding_stims.append(sf)

    n = len(stim_frames)
    return {
        "total_responses": responses,
        "total_stimuli": n,
        "response_rate": responses / n if n > 0 else 0.0,
        "responding_stim_frames": responding_stims,
    }


# =========================================================================
# Paired-Pulse Ratio (PPR)
# =========================================================================

def quantify_ppr(signal, frames, pulse1_frame, pulse2_frame,
                 response_window=4):
    """
    Quantify the Paired-Pulse Ratio for a single ROI.

    For each pulse, finds the maximum intensity within the response window
    (pulse_frame + 1 … pulse_frame + response_window).  Peaks outside both
    windows are classified as *asynchronous*.

    Parameters
    ----------
    signal : array-like
        Intensity values for one ROI trace.
    frames : array-like
        Corresponding frame numbers (sorted).
    pulse1_frame, pulse2_frame : int
        Frame numbers of the 1st and 2nd stimulus.
    response_window : int
        Frames after each pulse to look for a peak.

    Returns
    -------
    dict
        peak1_intensity, peak1_frame,
        peak2_intensity, peak2_frame,
        ppr (peak2 / peak1), n_async_peaks, async_frames
    """
    signal = np.asarray(signal, dtype=float)
    frames = np.asarray(frames)

    def _best_in_window(pulse_frame):
        lo = pulse_frame + 1
        hi = pulse_frame + response_window
        mask = (frames >= lo) & (frames <= hi)
        if not np.any(mask):
            return np.nan, np.nan
        idx = np.where(mask)[0]
        best = idx[np.argmax(signal[idx])]
        return float(signal[best]), int(frames[best])

    p1_val, p1_fr = _best_in_window(pulse1_frame)
    p2_val, p2_fr = _best_in_window(pulse2_frame)

    ppr = (p2_val / p1_val) if (p1_val and not np.isnan(p1_val)
                                 and p1_val != 0) else np.nan

    # Asynchronous peaks: local maxima outside both response windows
    from scipy.signal import argrelextrema as _arel
    if len(signal) >= 3:
        lm = _arel(signal, np.greater, order=1)[0]
    else:
        lm = np.array([], dtype=int)

    w1_lo, w1_hi = pulse1_frame + 1, pulse1_frame + response_window
    w2_lo, w2_hi = pulse2_frame + 1, pulse2_frame + response_window
    async_idx = [
        i for i in lm
        if not (w1_lo <= frames[i] <= w1_hi or w2_lo <= frames[i] <= w2_hi)
    ]

    return {
        "peak1_intensity": p1_val,
        "peak1_frame": p1_fr,
        "peak2_intensity": p2_val,
        "peak2_frame": p2_fr,
        "ppr": ppr,
        "n_async_peaks": len(async_idx),
        "async_frames": [int(frames[i]) for i in async_idx],
    }


# =========================================================================
# Main trace analysis
# =========================================================================

def analyze_combined_traces(combined_df, params):
    """
    Run firing detection on every ROI in the combined DataFrame.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Output of ``analyze_results.combine_csv_files``.
    params : dict
        sd_multiplier, rolling_window, baseline_start_frame,
        baseline_end_frame, order, frame_rate, stim_start_frame,
        response_window

    Returns
    -------
    combined_with_peaks : DataFrame
        *combined_df* with boolean ``accepted_peak`` column.
    firings_per_roi : DataFrame
        One row per ROI with firing counts, peak stats, and
        (for evoked) stimulus-response metrics.
    """
    sd_mult   = params.get("sd_multiplier", 4.0)
    roll_win  = params.get("rolling_window", 10)
    bl_start  = params.get("baseline_start_frame", 0)
    bl_end    = params.get("baseline_end_frame", 49)
    order     = params.get("order", 1)
    frame_rate    = params.get("frame_rate", 100.0)
    stim_start    = params.get("stim_start_frame", 50)
    resp_win      = params.get("response_window", 5)
    ppr_p1        = params.get("ppr_pulse1_frame", 50)
    ppr_p2        = params.get("ppr_pulse2_frame", 60)
    ppr_rw        = params.get("ppr_response_window", 4)

    # Prefer subtracted intensity
    sig_col = ("mean_intensity_subtracted"
               if "mean_intensity_subtracted" in combined_df.columns
               else "mean_intensity")

    combined_df = combined_df.copy()
    combined_df["accepted_peak"] = False

    # Collect original-index positions of peak rows for efficient bulk update
    peak_indices = []
    roi_rows = []

    grouped = combined_df.groupby(["image_name", "label"], sort=False)
    total = len(grouped)

    evoked_info_printed = set()

    for i, ((img, lbl), roi_df) in enumerate(grouped):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  Analyzing ROI {i + 1}/{total} ...")

        roi_sorted = roi_df.sort_values("slice")
        signal = roi_sorted[sig_col].values
        frames = roi_sorted["slice"].values

        if len(signal) < 3:
            continue

        det = detect_firings(signal, frames, sd_mult, roll_win,
                             baseline_start_frame=bl_start,
                             baseline_end_frame=bl_end,
                             order=order)

        # Record peak indices (original DataFrame index)
        if len(det["accepted_frames"]) > 0:
            accepted_set = set(det["accepted_frames"].tolist())
            matches = roi_sorted[roi_sorted["slice"].isin(accepted_set)]
            peak_indices.extend(matches.index.tolist())

        # Metadata
        first = roi_sorted.iloc[0]
        grp         = first.get("group", "unknown")
        exp_type    = first.get("experiment_type", "unknown")
        evk_sub     = first.get("evoked_subtype", None)
        n_ap        = first.get("action_potentials", None)
        stim_hz     = first.get("stimulus_frequency_hz", None)

        # Carry forward morphological / quality columns from the raw CSV
        roi_area = first.get("area", np.nan)
        major = first.get("axis_major_length", np.nan)
        minor = first.get("axis_minor_length", np.nan)
        try:
            aspect_ratio = float(major) / float(minor) if minor and float(minor) > 0 else np.nan
        except (TypeError, ValueError, ZeroDivisionError):
            aspect_ratio = np.nan
        roi_snr = first.get("snr", np.nan)
        roi_eccentricity = first.get("eccentricity", np.nan)
        roi_solidity = first.get("solidity", np.nan)
        roi_perimeter = first.get("perimeter", np.nan)
        roi_circularity = first.get("circularity", np.nan)

        info = {
            "image_name": img,
            "label": lbl,
            "group": grp,
            "experiment_type": exp_type,
            "evoked_subtype": evk_sub,
            "action_potentials": n_ap,
            "stimulus_frequency_hz": stim_hz,
            "total_firings": len(det["accepted_frames"]),
            "mean_peak_intensity": (
                float(np.mean(det["accepted_values"]))
                if len(det["accepted_values"]) > 0 else np.nan
            ),
            "max_peak_intensity": (
                float(np.max(det["accepted_values"]))
                if len(det["accepted_values"]) > 0 else np.nan
            ),
            # Baseline statistics
            "baseline_mean": det["baseline_mean"],
            "baseline_sd": det["baseline_sd"],
            "baseline_median": det["baseline_median"],
            "baseline_mad": det["baseline_mad"],
            "baseline_p5": det["baseline_p5"],
            "baseline_p95": det["baseline_p95"],
            "baseline_slope": det["baseline_slope"],
            # Whole-trace statistics
            "max_signal": float(np.nanmax(signal)),
            "signal_range": float(np.nanmax(signal) - np.nanmin(signal)),
            # Morphological / quality columns
            "area": roi_area,
            "aspect_ratio": aspect_ratio,
            "eccentricity": roi_eccentricity,
            "solidity": roi_solidity,
            "perimeter": roi_perimeter,
            "circularity": roi_circularity,
            "snr": roi_snr,
        }

        # --- Paired-Pulse Ratio ---
        if evk_sub == "paired_pulse":
            ppr_res = quantify_ppr(
                signal, frames, ppr_p1, ppr_p2,
                response_window=ppr_rw,
            )
            info["ppr_peak1_intensity"] = ppr_res["peak1_intensity"]
            info["ppr_peak1_frame"]     = ppr_res["peak1_frame"]
            info["ppr_peak2_intensity"] = ppr_res["peak2_intensity"]
            info["ppr_peak2_frame"]     = ppr_res["peak2_frame"]
            info["ppr"]                 = ppr_res["ppr"]
            info["ppr_n_async"]         = ppr_res["n_async_peaks"]

            if "ppr_printed" not in evoked_info_printed:
                print(f"  PPR config: pulse1={ppr_p1}, pulse2={ppr_p2}, "
                      f"window={ppr_rw}")
                evoked_info_printed.add("ppr_printed")

        # --- Train / generic evoked response quantification ---
        elif (exp_type == "evoked" and n_ap is not None
                and stim_hz is not None):
            try:
                n_ap_int = int(n_ap)
                stim_hz_f = float(stim_hz)
                sf, si = compute_stim_frames(
                    n_ap_int, stim_hz_f, frame_rate, stim_start
                )
                ev = quantify_evoked_responses(
                    det["accepted_frames"], sf, resp_win
                )
                info["stim_responses"] = ev["total_responses"]
                info["stim_total"]     = ev["total_stimuli"]
                info["response_rate"]  = ev["response_rate"]

                key = (n_ap_int, stim_hz_f)
                if key not in evoked_info_printed:
                    last_sf = sf[-1] if sf else stim_start
                    print(f"  Evoked config: {n_ap_int}AP @ {stim_hz_f}Hz, "
                          f"interval={si} frames, "
                          f"last stim frame={last_sf}")
                    evoked_info_printed.add(key)
            except (ValueError, ZeroDivisionError):
                pass

        roi_rows.append(info)

    # Bulk-set accepted peaks
    if peak_indices:
        combined_df.loc[peak_indices, "accepted_peak"] = True

    firings_per_roi = pd.DataFrame(roi_rows)
    n_peaks = int(combined_df["accepted_peak"].sum())
    print(f"  Analysis complete: {len(firings_per_roi)} ROIs, "
          f"{n_peaks} accepted peaks")

    return combined_df, firings_per_roi


# =========================================================================
# Outlier detection
# =========================================================================

# All supported outlier metrics: (source_col, z_col_name)
OUTLIER_METRICS = [
    ("baseline_median", "z_baseline_median"),
    ("baseline_sd",     "z_baseline_sd"),
    ("max_signal",      "z_max_signal"),
    ("total_firings",   "z_total_firings"),
    ("area",            "z_area"),
    ("solidity",        "z_solidity"),
    ("circularity",     "z_circularity"),
]

# Default outlier config (used when caller passes None or legacy threshold)
DEFAULT_OUTLIER_CONFIG = {
    "baseline_median": {"enabled": True,  "threshold": 3.0},
    "baseline_sd":     {"enabled": True,  "threshold": 3.0},
    "max_signal":      {"enabled": False, "threshold": 3.0},
    "total_firings":   {"enabled": True,  "threshold": 3.0},
    "area":            {"enabled": True,  "threshold": 3.0},
    "solidity":        {"enabled": True,  "threshold": 3.0},
    "circularity":     {"enabled": True,  "threshold": 3.0},
}


def compute_outlier_scores(firings_per_roi, outlier_config=None,
                           threshold=None):
    """Compute per-metric outlier z-scores and flag ROIs.

    Within each image the modified z-score (MAD-based) of each metric
    is computed.  An ROI is flagged if **any enabled metric** exceeds
    its own threshold.  The ``outlier_reasons`` column lists which
    metric(s) triggered the flag.

    Parameters
    ----------
    firings_per_roi : pd.DataFrame
        Must contain at least ``image_name`` and ``label`` columns.
    outlier_config : dict or None
        Per-metric configuration, e.g.::

            {"baseline_median": {"enabled": True, "threshold": 3.0},
             "area":            {"enabled": False, "threshold": 3.0},
             ...}

        If *None*, ``DEFAULT_OUTLIER_CONFIG`` is used.
    threshold : float or None
        **Legacy parameter.**  If provided (and *outlier_config* is None),
        a single threshold is applied to all default-enabled metrics.

    Returns
    -------
    pd.DataFrame
        *firings_per_roi* with additional columns:
        per-metric ``z_*`` scores, ``outlier_flag`` (bool), and
        ``outlier_reasons`` (comma-separated string of metric names
        that exceeded their threshold).
    """
    # Resolve config
    if outlier_config is None:
        cfg = {k: dict(v) for k, v in DEFAULT_OUTLIER_CONFIG.items()}
        if threshold is not None:
            for v in cfg.values():
                v["threshold"] = float(threshold)
    else:
        cfg = {k: dict(v) for k, v in DEFAULT_OUTLIER_CONFIG.items()}
        for k, v in outlier_config.items():
            if k in cfg:
                cfg[k].update(v)

    df = firings_per_roi.copy()

    # Initialise z-score columns
    z_cols = []
    for src, zcol in OUTLIER_METRICS:
        df[zcol] = np.nan
        z_cols.append(zcol)

    # Compute z-scores per image
    for img, idx in df.groupby("image_name").groups.items():
        sub = df.loc[idx]
        if len(sub) < 3:
            for _, zcol in OUTLIER_METRICS:
                df.loc[idx, zcol] = 0.0
            continue

        for src, zcol in OUTLIER_METRICS:
            if src not in sub.columns:
                df.loc[idx, zcol] = 0.0
                continue
            vals = pd.to_numeric(sub[src], errors="coerce")
            med = vals.median()
            mad = np.median(np.abs(vals - med))
            if mad == 0 or np.isnan(mad):
                sd = vals.std()
                if sd == 0 or np.isnan(sd):
                    df.loc[idx, zcol] = 0.0
                else:
                    df.loc[idx, zcol] = ((vals - med) / sd).abs().values
            else:
                df.loc[idx, zcol] = (
                    (vals - med) / (mad / 0.6745)
                ).abs().values

    # Per-row flagging: check each enabled metric against its threshold
    flag_series = pd.Series(False, index=df.index)
    reasons_series = pd.Series("", index=df.index, dtype=str)

    for src, zcol in OUTLIER_METRICS:
        mcfg = cfg.get(src, {})
        if not mcfg.get("enabled", False):
            continue
        thr = float(mcfg.get("threshold", 3.0))
        exceeded = df[zcol] >= thr
        flag_series = flag_series | exceeded
        # Append metric name to reasons where exceeded
        reasons_series = reasons_series.where(
            ~exceeded,
            reasons_series + src + ", "
        )

    # Clean trailing ", "
    reasons_series = reasons_series.str.rstrip(", ")

    df["outlier_flag"] = flag_series
    df["outlier_reasons"] = reasons_series

    return df


# =========================================================================
# Summary generators
# =========================================================================

def generate_per_image_summary(firings_per_roi):
    """One row per image with aggregated firing statistics."""
    if firings_per_roi is None or len(firings_per_roi) == 0:
        return pd.DataFrame()

    gc = ["image_name", "group", "experiment_type",
          "action_potentials", "stimulus_frequency_hz"]
    gc = [c for c in gc if c in firings_per_roi.columns]

    agg_dict = {
        "label": "nunique",
        "total_firings": ["sum", "mean", "std", "median"],
        "mean_peak_intensity": "mean",
        "max_peak_intensity": "max",
    }

    per_img = firings_per_roi.groupby(gc, dropna=False).agg(agg_dict)
    per_img.columns = ["_".join(c).rstrip("_") for c in per_img.columns]
    per_img = per_img.rename(columns={
        "label_nunique": "n_rois",
        "total_firings_sum": "total_firings",
        "total_firings_mean": "mean_firings_per_roi",
        "total_firings_std": "sd_firings_per_roi",
        "total_firings_median": "median_firings_per_roi",
        "mean_peak_intensity_mean": "mean_peak_intensity",
        "max_peak_intensity_max": "max_peak_intensity",
    }).reset_index()

    # Baseline statistics per image (aggregate ROI-level baseline stats)
    bl_cols = {
        "baseline_median": ["mean", "std"],
        "baseline_mad": ["mean", "std"],
        "baseline_mean": ["mean"],
        "baseline_sd": ["mean"],
        "baseline_slope": ["mean", "std"],
    }
    bl_agg = {}
    for col, funcs in bl_cols.items():
        if col in firings_per_roi.columns:
            bl_agg[col] = funcs
    if bl_agg:
        bl_img = firings_per_roi.groupby(gc, dropna=False).agg(bl_agg)
        bl_img.columns = ["_".join(c) for c in bl_img.columns]
        bl_img = bl_img.reset_index()
        per_img = per_img.merge(bl_img, on=gc, how="left")

    # Outlier counts per image
    if "outlier_flag" in firings_per_roi.columns:
        outlier_agg = (
            firings_per_roi.groupby(gc, dropna=False)
            .agg(
                n_flagged_rois=("outlier_flag", "sum"),
                _n_total=("outlier_flag", "count"),
            )
            .reset_index()
        )
        outlier_agg["pct_flagged_rois"] = (
            outlier_agg["n_flagged_rois"] / outlier_agg["_n_total"] * 100
        ).round(1)
        outlier_agg = outlier_agg.drop(columns=["_n_total"])
        per_img = per_img.merge(outlier_agg, on=gc, how="left")

    # Evoked response rate per image
    if "response_rate" in firings_per_roi.columns:
        ev = (firings_per_roi.dropna(subset=["response_rate"])
              .groupby(gc, dropna=False)
              .agg(mean_response_rate=("response_rate", "mean"),
                   sd_response_rate=("response_rate", "std"),
                   median_response_rate=("response_rate", "median"))
              .reset_index())
        per_img = per_img.merge(ev, on=gc, how="left")

    return per_img


def generate_group_summary(firings_per_roi):
    """One row per group (+condition) with mean / SD / median.

    Statistics are computed over **per-image means** (each image is one
    data-point) rather than individual ROIs, so that images with more
    ROIs do not dominate the summary.
    """
    if firings_per_roi is None or len(firings_per_roi) == 0:
        return pd.DataFrame()

    gc = ["group", "experiment_type",
          "action_potentials", "stimulus_frequency_hz"]
    gc = [c for c in gc if c in firings_per_roi.columns]

    # Step 1 – aggregate to per-image means
    img_gc = gc + ["image_name"]
    value_cols = ["total_firings", "mean_peak_intensity"]
    if "response_rate" in firings_per_roi.columns:
        value_cols.append("response_rate")
    # Add baseline cols if present
    for bcol in ["baseline_median", "baseline_mad", "baseline_slope"]:
        if bcol in firings_per_roi.columns:
            value_cols.append(bcol)

    agg_dict = {c: "mean" for c in value_cols if c in firings_per_roi.columns}
    agg_dict["label"] = "count"  # keep n_rois for reference
    # Outlier flag: count flagged per image
    if "outlier_flag" in firings_per_roi.columns:
        agg_dict["outlier_flag"] = "sum"
    per_image = (
        firings_per_roi.groupby(img_gc, dropna=False)
        .agg(agg_dict)
        .rename(columns={"label": "n_rois", "outlier_flag": "n_flagged"})
        .reset_index()
    )

    # Step 2 – group-level stats from per-image means
    base_agg = dict(
        n_images=("image_name", "nunique"),
        total_rois=("n_rois", "sum"),
        mean_firings=("total_firings", "mean"),
        sd_firings=("total_firings", "std"),
        median_firings=("total_firings", "median"),
        mean_peak_intensity=("mean_peak_intensity", "mean"),
        sd_peak_intensity=("mean_peak_intensity", "std"),
    )

    # Baseline stats from per-image means
    if "baseline_median" in per_image.columns:
        base_agg["baseline_median_mean"] = ("baseline_median", "mean")
        base_agg["baseline_median_sd"] = ("baseline_median", "std")
    if "baseline_mad" in per_image.columns:
        base_agg["baseline_mad_mean"] = ("baseline_mad", "mean")
        base_agg["baseline_mad_sd"] = ("baseline_mad", "std")
    if "baseline_slope" in per_image.columns:
        base_agg["baseline_slope_mean"] = ("baseline_slope", "mean")
        base_agg["baseline_slope_sd"] = ("baseline_slope", "std")

    # Outlier counts
    if "n_flagged" in per_image.columns:
        base_agg["total_flagged_rois"] = ("n_flagged", "sum")

    summary = (
        per_image.groupby(gc, dropna=False)
        .agg(**base_agg)
        .reset_index()
    )

    # Compute pct_flagged at group level
    if "total_flagged_rois" in summary.columns and "total_rois" in summary.columns:
        summary["pct_flagged_rois"] = (
            summary["total_flagged_rois"] / summary["total_rois"] * 100
        ).round(1)

    if "response_rate" in per_image.columns:
        ev = (per_image.dropna(subset=["response_rate"])
              .groupby(gc, dropna=False)
              .agg(mean_response_rate=("response_rate", "mean"),
                   sd_response_rate=("response_rate", "std"),
                   median_response_rate=("response_rate", "median"))
              .reset_index())
        summary = summary.merge(ev, on=gc, how="left")

    return summary


def generate_ppr_image_summary(firings_per_roi):
    """Per-image PPR summary: mean peak1, mean peak2, mean PPR ratio.

    Returns a DataFrame with one row per image (PPR images only).
    """
    ppr_df = firings_per_roi.dropna(subset=["ppr"])
    if len(ppr_df) == 0:
        return pd.DataFrame()

    gc = ["image_name", "group", "experiment_type", "evoked_subtype",
          "stimulus_frequency_hz"]
    gc = [c for c in gc if c in ppr_df.columns]

    summary = (
        ppr_df.groupby(gc, dropna=False)
        .agg(
            n_rois=("label", "count"),
            mean_peak1_intensity=("ppr_peak1_intensity", "mean"),
            mean_peak2_intensity=("ppr_peak2_intensity", "mean"),
            mean_ppr=("ppr", "mean"),
            sd_ppr=("ppr", "std"),
            median_ppr=("ppr", "median"),
            mean_async_events=("ppr_n_async", "mean"),
        )
        .reset_index()
    )
    # Image-level PPR from image-level means (ratio of means)
    summary["ppr_from_means"] = (
        summary["mean_peak2_intensity"] / summary["mean_peak1_intensity"]
    )
    return summary


def generate_ppr_group_summary(ppr_image_summary):
    """Group-level PPR summary from per-image means."""
    if ppr_image_summary is None or len(ppr_image_summary) == 0:
        return pd.DataFrame()

    gc = ["group", "experiment_type", "evoked_subtype",
          "stimulus_frequency_hz"]
    gc = [c for c in gc if c in ppr_image_summary.columns]

    return (
        ppr_image_summary.groupby(gc, dropna=False)
        .agg(
            n_images=("image_name", "nunique"),
            total_rois=("n_rois", "sum"),
            mean_peak1=("mean_peak1_intensity", "mean"),
            sd_peak1=("mean_peak1_intensity", "std"),
            mean_peak2=("mean_peak2_intensity", "mean"),
            sd_peak2=("mean_peak2_intensity", "std"),
            mean_ppr=("mean_ppr", "mean"),
            sd_ppr=("mean_ppr", "std"),
            median_ppr=("mean_ppr", "median"),
            mean_ppr_from_means=("ppr_from_means", "mean"),
        )
        .reset_index()
    )


# =========================================================================
# Plotting helpers
# =========================================================================

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _group_colors(groups):
    return {g: _COLORS[i % len(_COLORS)] for i, g in enumerate(groups)}


def _present_groups(df, col, groups):
    """Return only groups that actually appear in the data."""
    present = set(df[col].str.lower().unique())
    return [g for g in groups if g.lower() in present]


def _safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


# ---- Spontaneous combined traces ----

def plot_combined_traces(combined_df, groups, output_dir):
    """Overlay all ROI traces per group with mean +/- SEM."""
    sig_col = ("mean_intensity_subtracted"
               if "mean_intensity_subtracted" in combined_df.columns
               else "mean_intensity")
    grps = _present_groups(combined_df, "group", groups)
    colors = _group_colors(grps)

    def _collect_traces(gdf, max_frame):
        traces = []
        for (_, _), rdf in gdf.groupby(["image_name", "label"]):
            rdf = rdf.sort_values("slice")
            t = np.full(max_frame, np.nan)
            fr = rdf["slice"].values.astype(int)
            v = rdf[sig_col].values
            ok = (fr >= 0) & (fr < max_frame)
            t[fr[ok]] = v[ok]
            traces.append(t)
        return traces

    # Per-group plots
    for grp in grps:
        gdf = combined_df[combined_df["group"].str.lower() == grp.lower()]
        if len(gdf) == 0:
            continue
        max_frame = int(gdf["slice"].max()) + 1
        traces = _collect_traces(gdf, max_frame)
        if not traces:
            continue

        # Scale opacity by number of traces so individual lines stay visible
        n_tr = len(traces)
        trace_alpha = max(0.15, min(0.6, 10.0 / n_tr))

        fig, ax = plt.subplots(figsize=(14, 5))
        for tr in traces:
            ax.plot(np.arange(max_frame), tr,
                    color=colors[grp], alpha=trace_alpha, linewidth=0.5)
        arr = np.array(traces)
        mean_t = np.nanmean(arr, axis=0)
        n_valid = np.sum(~np.isnan(arr), axis=0).clip(1)
        sem_t = np.nanstd(arr, axis=0) / np.sqrt(n_valid)
        x = np.arange(max_frame)
        ax.plot(x, mean_t, color=colors[grp], linewidth=2,
                label=f"{grp} mean (n={n_tr})")
        ax.fill_between(x, mean_t - sem_t, mean_t + sem_t,
                        color=colors[grp], alpha=0.25)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Spontaneous Traces \u2014 {grp} ({len(traces)} ROIs)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"traces_{grp}.png"), dpi=150)
        plt.close(fig)

    # All groups on one plot — raw traces, no averaging
    if len(grps) > 1:
        fig, ax = plt.subplots(figsize=(14, 5))
        for grp in grps:
            gdf = combined_df[combined_df["group"].str.lower() == grp.lower()]
            if len(gdf) == 0:
                continue
            max_frame = int(gdf["slice"].max()) + 1
            traces = _collect_traces(gdf, max_frame)
            if not traces:
                continue
            n_tr = len(traces)
            trace_alpha = max(0.25, min(0.7, 15.0 / n_tr))
            x = np.arange(max_frame)
            for i, tr in enumerate(traces):
                ax.plot(x, tr, color=colors[grp], alpha=trace_alpha,
                        linewidth=1.0,
                        label=f"{grp} (n={n_tr})" if i == 0 else None)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Intensity")
        ax.set_title("Spontaneous Traces \u2014 All Groups (raw)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "traces_all_groups.png"), dpi=150)
        plt.close(fig)


# ---- Aggregation helpers ----

def _per_image_means(firings_per_roi, value_col):
    """Aggregate *value_col* to one mean value per image (the statistical
    unit for between-group comparisons)."""
    return (
        firings_per_roi
        .groupby(["image_name", "group"], dropna=False)[value_col]
        .mean()
        .reset_index()
    )


def _per_image_peak_means(combined_df, sig_col):
    """Mean accepted-peak intensity per image."""
    peaks = combined_df[combined_df["accepted_peak"] == True]
    if len(peaks) == 0:
        return pd.DataFrame(columns=["image_name", "group", sig_col])
    return (
        peaks
        .groupby(["image_name", "group"], dropna=False)[sig_col]
        .mean()
        .reset_index()
    )


def _draw_boxplot(ax, data, grps, colors, ylabel):
    """Draw a boxplot with individual image-level data points."""
    bp = ax.boxplot(data, labels=grps, patch_artist=True, widths=0.5)
    for patch, g in zip(bp["boxes"], grps):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.7)
    for i, (d, g) in enumerate(zip(data, grps)):
        if len(d) > 0:
            jitter = np.random.default_rng(42).normal(0, 0.04, size=len(d))
            ax.scatter(np.full_like(d, i + 1, dtype=float) + jitter, d,
                       color=colors[g], alpha=0.6, s=30, zorder=3,
                       edgecolors="black", linewidths=0.5)
    ax.set_ylabel(ylabel)


# ---- Box plots (per-image means as data points) ----

def plot_firings_boxplot(firings_per_roi, groups, output_dir,
                         condition_label=""):
    """Boxplot of mean firings per image, grouped by group."""
    grps = _present_groups(firings_per_roi, "group", groups)
    if not grps:
        return
    colors = _group_colors(grps)

    img_means = _per_image_means(firings_per_roi, "total_firings")
    data = [
        img_means[img_means["group"].str.lower() == g.lower()
                  ]["total_firings"].dropna().values
        for g in grps
    ]

    fig, ax = plt.subplots(figsize=(max(4, len(grps) * 1.8), 5))
    _draw_boxplot(ax, data, grps, colors, "Mean Firings per ROI (per image)")

    title = "Firings per Image"
    if condition_label:
        title += f" \u2014 {condition_label}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "firings_boxplot_by_group.png"),
                dpi=150)
    plt.close(fig)


# ---- Histograms (side-by-side subplots, free y-axis) ----

def plot_peak_intensity_histogram(combined_df, groups, output_dir,
                                  condition_label=""):
    """Side-by-side histograms of accepted-peak intensities per group,
    each with its own y-axis scale for easy visual comparison."""
    peaks = combined_df[combined_df["accepted_peak"] == True]
    if len(peaks) == 0:
        return

    sig_col = ("mean_intensity_subtracted"
               if "mean_intensity_subtracted" in peaks.columns
               else "mean_intensity")

    grps = _present_groups(peaks, "group", groups)
    if not grps:
        return
    colors = _group_colors(grps)

    # Shared x-range across all groups
    all_vals = peaks[sig_col].dropna().values
    if len(all_vals) == 0:
        return
    x_lo, x_hi = np.nanmin(all_vals), np.nanmax(all_vals)
    bins = np.linspace(x_lo, x_hi, 31)

    n_grps = len(grps)
    fig, axes = plt.subplots(1, n_grps, figsize=(5 * n_grps, 4),
                             sharey=False, squeeze=False)
    axes = axes.ravel()

    for idx, g in enumerate(grps):
        ax = axes[idx]
        vals = peaks[peaks["group"].str.lower() == g.lower()
                     ][sig_col].dropna().values
        if len(vals) > 0:
            ax.hist(vals, bins=bins, color=colors[g], edgecolor="black",
                    linewidth=0.5, alpha=0.8)
        ax.set_title(f"{g} (n={len(vals)})")
        ax.set_xlabel("Peak Intensity")
        if idx == 0:
            ax.set_ylabel("Count")

    suptitle = "Peak Intensity Distribution"
    if condition_label:
        suptitle += f" \u2014 {condition_label}"
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "peak_intensity_histogram.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_peak_intensity_boxplot(combined_df, groups, output_dir,
                                condition_label=""):
    """Boxplot of mean accepted-peak intensity per image, grouped by group."""
    peaks = combined_df[combined_df["accepted_peak"] == True]
    if len(peaks) == 0:
        return

    sig_col = ("mean_intensity_subtracted"
               if "mean_intensity_subtracted" in peaks.columns
               else "mean_intensity")

    grps = _present_groups(peaks, "group", groups)
    if not grps:
        return
    colors = _group_colors(grps)

    img_means = _per_image_peak_means(combined_df, sig_col)
    data = [
        img_means[img_means["group"].str.lower() == g.lower()
                  ][sig_col].dropna().values
        for g in grps
    ]

    fig, ax = plt.subplots(figsize=(max(4, len(grps) * 1.8), 5))
    _draw_boxplot(ax, data, grps, colors,
                  "Mean Peak Intensity (per image)")

    title = "Peak Intensity by Group"
    if condition_label:
        title += f" \u2014 {condition_label}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "peak_intensity_boxplot_by_group.png"),
        dpi=150,
    )
    plt.close(fig)


def plot_response_rate_boxplot(firings_per_roi, groups, output_dir,
                               condition_label=""):
    """Boxplot of mean evoked response rate per image, grouped by group."""
    df = firings_per_roi.dropna(subset=["response_rate"])
    if len(df) == 0:
        return

    grps = _present_groups(df, "group", groups)
    if not grps:
        return
    colors = _group_colors(grps)

    img_means = _per_image_means(df, "response_rate")
    data = [
        img_means[img_means["group"].str.lower() == g.lower()
                  ]["response_rate"].dropna().values
        for g in grps
    ]

    fig, ax = plt.subplots(figsize=(max(4, len(grps) * 1.8), 5))
    _draw_boxplot(ax, data, grps, colors,
                  "Mean Response Rate (per image)")

    title = "Stimulus Response Rate"
    if condition_label:
        title += f" \u2014 {condition_label}"
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "response_rate_boxplot_by_group.png"),
        dpi=150,
    )
    plt.close(fig)


def plot_ppr_boxplot(ppr_image_summary, groups, output_dir,
                     condition_label=""):
    """Boxplot of per-image PPR values, grouped by group."""
    if ppr_image_summary is None or len(ppr_image_summary) == 0:
        return
    grps = _present_groups(ppr_image_summary, "group", groups)
    if not grps:
        return
    colors = _group_colors(grps)

    data = [
        ppr_image_summary[
            ppr_image_summary["group"].str.lower() == g.lower()
        ]["mean_ppr"].dropna().values
        for g in grps
    ]

    fig, axes = plt.subplots(1, 3, figsize=(max(6, len(grps) * 4.5), 5))

    # Panel 1: PPR ratio
    _draw_boxplot(axes[0], data, grps, colors, "PPR (peak2 / peak1)")
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("Paired-Pulse Ratio")

    # Panel 2: Mean peak1 intensity per image
    data1 = [
        ppr_image_summary[
            ppr_image_summary["group"].str.lower() == g.lower()
        ]["mean_peak1_intensity"].dropna().values
        for g in grps
    ]
    _draw_boxplot(axes[1], data1, grps, colors, "Intensity (1st pulse)")
    axes[1].set_title("1st Pulse Response")

    # Panel 3: Mean peak2 intensity per image
    data2 = [
        ppr_image_summary[
            ppr_image_summary["group"].str.lower() == g.lower()
        ]["mean_peak2_intensity"].dropna().values
        for g in grps
    ]
    _draw_boxplot(axes[2], data2, grps, colors, "Intensity (2nd pulse)")
    axes[2].set_title("2nd Pulse Response")

    suptitle = "Paired-Pulse Analysis"
    if condition_label:
        suptitle += f" \u2014 {condition_label}"
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ppr_boxplot_by_group.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- Baseline sanity check ----

def plot_baseline_boxplot(firings_per_roi, groups, output_dir):
    """Two-panel boxplot: per-image median baseline level + MAD spread.

    Each dot = one image (mean of per-ROI values for that image).
    Used as a sanity check that baseline does not shift across images/groups.
    """
    grps = _present_groups(firings_per_roi, "group", groups)
    if not grps:
        return
    colors = _group_colors(grps)

    img_bl = _per_image_means(firings_per_roi, "baseline_median")
    has_mad = "baseline_mad" in firings_per_roi.columns
    if has_mad:
        img_mad = _per_image_means(firings_per_roi, "baseline_mad")
    has_slope = "baseline_slope" in firings_per_roi.columns
    if has_slope:
        img_slope = _per_image_means(firings_per_roi, "baseline_slope")

    n_panels = 1 + int(has_mad) + int(has_slope)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(max(5, len(grps) * 2) * n_panels, 5),
                             squeeze=False)
    axes = axes.ravel()

    # Panel 1: Baseline median level
    data_bl = [
        img_bl[img_bl["group"].str.lower() == g.lower()
               ]["baseline_median"].dropna().values
        for g in grps
    ]
    _draw_boxplot(axes[0], data_bl, grps, colors,
                  "Median Baseline Intensity (per image)")
    axes[0].set_title("Baseline Level")

    panel = 1
    # Panel 2: Baseline MAD (variability)
    if has_mad:
        data_mad = [
            img_mad[img_mad["group"].str.lower() == g.lower()
                    ]["baseline_mad"].dropna().values
            for g in grps
        ]
        _draw_boxplot(axes[panel], data_mad, grps, colors,
                      "Median Baseline MAD (per image)")
        axes[panel].set_title("Baseline Variability (MAD)")
        panel += 1

    # Panel 3: Baseline slope (drift)
    if has_slope:
        data_slope = [
            img_slope[img_slope["group"].str.lower() == g.lower()
                      ]["baseline_slope"].dropna().values
            for g in grps
        ]
        _draw_boxplot(axes[panel], data_slope, grps, colors,
                      "Mean Baseline Slope (per image)")
        axes[panel].set_title("Baseline Drift (Slope)")
        axes[panel].axhline(0, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle("Baseline Sanity Check", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "baseline_sanity_check.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- Outlier summary plots ----

def plot_outlier_summary(firings_per_roi, per_image_summary, groups,
                         output_dir):
    """Two-panel outlier summary.

    Panel 1: Bar chart of % flagged ROIs per image, grouped by group.
    Panel 2: Scatter of max z-score (across all metrics) vs ROI index,
             colored by group. Flagged ROIs are highlighted.
    """
    if "outlier_flag" not in firings_per_roi.columns:
        return

    grps = _present_groups(firings_per_roi, "group", groups)
    if not grps:
        return
    colors = _group_colors(grps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: % flagged per image (bar chart) ---
    ax1 = axes[0]
    if (per_image_summary is not None and len(per_image_summary) > 0
            and "pct_flagged_rois" in per_image_summary.columns):
        pi = per_image_summary.sort_values(
            ["group", "image_name"]).reset_index(drop=True)
        bar_colors = []
        for _, row in pi.iterrows():
            g = row.get("group", "")
            matched = [k for k in grps if k.lower() == str(g).lower()]
            bar_colors.append(colors[matched[0]] if matched else "#888888")
        x = np.arange(len(pi))
        ax1.bar(x, pi["pct_flagged_rois"].values, color=bar_colors,
                edgecolor="black", linewidth=0.5, alpha=0.8)
        ax1.set_xticks(x)
        labels = [str(n)[:20] for n in pi["image_name"].values]
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax1.set_ylabel("% Flagged ROIs")
        ax1.set_title("Outlier ROIs per Image")
        ax1.axhline(30, color="red", linestyle="--", linewidth=0.8,
                     label="30% warning")
        ax1.legend(fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No per-image data", transform=ax1.transAxes,
                 ha="center", va="center")

    # --- Panel 2: Max z-score scatter ---
    ax2 = axes[1]
    # Compute the max z across all z_* columns for the scatter y-axis
    z_cols = [zcol for _, zcol in OUTLIER_METRICS if zcol in firings_per_roi.columns]
    if z_cols:
        max_z = firings_per_roi[z_cols].max(axis=1).fillna(0.0)
    else:
        max_z = pd.Series(0.0, index=firings_per_roi.index)

    for g in grps:
        mask = firings_per_roi["group"].str.lower() == g.lower()
        sub_z = max_z[mask]
        if len(sub_z) == 0:
            continue
        ax2.scatter(sub_z.index, sub_z, color=colors[g],
                    alpha=0.4, s=8, label=g)
    # Highlight flagged ROIs
    flagged = firings_per_roi["outlier_flag"] == True
    if flagged.any():
        ax2.scatter(max_z[flagged].index, max_z[flagged],
                    facecolors="none", edgecolors="red", s=30,
                    linewidths=0.8, label="flagged", zorder=4)
    ax2.set_xlabel("ROI index")
    ax2.set_ylabel("Max z-score (across metrics)")
    ax2.set_title("Outlier z-Score Distribution")
    ax2.legend(fontsize=8)

    fig.suptitle("Outlier Summary", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "outlier_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Generate all plots
# =========================================================================

def generate_all_plots(combined_df, firings_per_roi, groups, plots_dir,
                       ppr_image_summary=None, per_image_summary=None):
    """Generate all analysis plots, organized into subdirectories."""
    _safe_mkdir(plots_dir)

    # ---------- Baseline & Outlier (all experiment types) ----------
    if "baseline_median" in firings_per_roi.columns:
        print("  Generating baseline sanity-check plot ...")
        plot_baseline_boxplot(firings_per_roi, groups, plots_dir)
    if "outlier_flag" in firings_per_roi.columns:
        print("  Generating outlier summary plot ...")
        plot_outlier_summary(firings_per_roi, per_image_summary, groups,
                             plots_dir)

    # ---------- Spontaneous ----------
    spon_comb = combined_df[
        combined_df["experiment_type"].str.lower() == "spontaneous"
    ]
    spon_roi = firings_per_roi[
        firings_per_roi["experiment_type"].str.lower() == "spontaneous"
    ]

    if len(spon_roi) > 0:
        spon_dir = os.path.join(plots_dir, "spontaneous")
        _safe_mkdir(spon_dir)
        print("  Generating spontaneous plots ...")
        plot_combined_traces(spon_comb, groups, spon_dir)
        plot_firings_boxplot(spon_roi, groups, spon_dir, "Spontaneous")
        plot_peak_intensity_histogram(spon_comb, groups, spon_dir,
                                      "Spontaneous")
        plot_peak_intensity_boxplot(spon_comb, groups, spon_dir,
                                    "Spontaneous")

    # ---------- Evoked ----------
    evoked_comb = combined_df[
        combined_df["experiment_type"].str.lower() == "evoked"
    ]
    evoked_roi = firings_per_roi[
        firings_per_roi["experiment_type"].str.lower() == "evoked"
    ]

    if len(evoked_roi) > 0:
        cond_cols = ["action_potentials", "stimulus_frequency_hz"]
        conditions = (
            evoked_roi[cond_cols].dropna().drop_duplicates()
        )

        for _, row in conditions.iterrows():
            n_ap = row["action_potentials"]
            hz = row["stimulus_frequency_hz"]
            label = f"{int(n_ap)}AP_{int(hz)}Hz"
            cond_dir = os.path.join(plots_dir, "evoked", label)
            _safe_mkdir(cond_dir)

            mask_roi = (
                (evoked_roi["action_potentials"] == n_ap)
                & (evoked_roi["stimulus_frequency_hz"] == hz)
            )
            mask_df = (
                (evoked_comb["action_potentials"] == n_ap)
                & (evoked_comb["stimulus_frequency_hz"] == hz)
            )

            cond_roi = evoked_roi[mask_roi]
            cond_df = evoked_comb[mask_df]

            print(f"  Generating evoked plots for {label} ...")
            plot_firings_boxplot(cond_roi, groups, cond_dir, label)
            plot_peak_intensity_histogram(cond_df, groups, cond_dir, label)
            plot_peak_intensity_boxplot(cond_df, groups, cond_dir, label)
            if "response_rate" in cond_roi.columns:
                plot_response_rate_boxplot(cond_roi, groups, cond_dir, label)

    # ---------- Paired-Pulse ----------
    if ppr_image_summary is not None and len(ppr_image_summary) > 0:
        ppr_dir = os.path.join(plots_dir, "paired_pulse")
        _safe_mkdir(ppr_dir)
        print("  Generating PPR plots ...")
        # Group PPR by Hz (different PPR frequencies)
        if "stimulus_frequency_hz" in ppr_image_summary.columns:
            for hz_val in ppr_image_summary["stimulus_frequency_hz"].dropna().unique():
                sub = ppr_image_summary[
                    ppr_image_summary["stimulus_frequency_hz"] == hz_val]
                label = f"PPR_{int(hz_val)}Hz"
                hz_dir = os.path.join(ppr_dir, label)
                _safe_mkdir(hz_dir)
                plot_ppr_boxplot(sub, groups, hz_dir, label)
        # Also an overall PPR plot
        plot_ppr_boxplot(ppr_image_summary, groups, ppr_dir, "PPR (all)")


# =========================================================================
# Orchestrator
# =========================================================================

def run_full_analysis(output_folder, params, progress_callback=None):
    """
    Full pipeline: combine CSVs -> detect firings -> summaries -> plots.

    Parameters
    ----------
    output_folder : str
        Root output directory.
    params : dict
        sd_multiplier, rolling_window, baseline_start_frame,
        baseline_end_frame, order, frame_rate, stim_start_frame,
        response_window, groups (list of str)
    progress_callback : callable or None
        Called with status strings.

    Returns
    -------
    dict or None
        Keys: combined_with_peaks, firings_per_roi, per_image,
              group_summary, analysis_dir
    """
    def _msg(s):
        print(s)
        if progress_callback:
            progress_callback(s)

    groups     = params.get("groups", ["WT", "APOE"])
    frame_rate = params.get("frame_rate", 100.0)

    # ---- Step 1: combine CSVs ----
    _msg("Step 1/4: Discovering and combining CSV files ...")
    from analyze_results import find_snr_labeled_csvs, combine_csv_files

    csv_files = find_snr_labeled_csvs(output_folder)
    if not csv_files:
        _msg("ERROR: No _full_SNRlabeled_ CSV files found.")
        return None

    _msg(f"  Found {len(csv_files)} CSV file(s)")
    combined_df = combine_csv_files(csv_files, frame_rate=frame_rate,
                                     custom_groups=groups)

    if combined_df is None or len(combined_df) == 0:
        _msg("ERROR: Combined DataFrame is empty.")
        return None

    _msg(f"  Combined: {len(combined_df)} rows, "
         f"{combined_df['image_name'].nunique()} images")

    # ---- Step 2: firing detection ----
    _msg("Step 2/4: Running firing detection ...")
    combined_with_peaks, firings_per_roi = analyze_combined_traces(
        combined_df, params
    )

    # ---- Step 2b: outlier scoring ----
    outlier_cfg = params.get("outlier_config", None)
    # Backward-compat: if old-style single threshold is present, use it
    outlier_thr = params.get("outlier_threshold", None)
    enabled_names = []
    if outlier_cfg:
        enabled_names = [k for k, v in outlier_cfg.items()
                         if v.get("enabled")]
        _msg(f"  Outlier detection: {len(enabled_names)} metric(s) enabled "
             f"({', '.join(enabled_names)})")
    else:
        _msg(f"  Outlier detection: using defaults "
             f"(threshold={outlier_thr or 3.0})")
    firings_per_roi = compute_outlier_scores(
        firings_per_roi, outlier_config=outlier_cfg, threshold=outlier_thr)
    n_flagged = int(firings_per_roi["outlier_flag"].sum())
    _msg(f"  Flagged {n_flagged} / {len(firings_per_roi)} ROIs as outliers")

    # ---- Step 3: summaries ----
    _msg("Step 3/5: Generating summary tables ...")
    per_image     = generate_per_image_summary(firings_per_roi)
    group_summary = generate_group_summary(firings_per_roi)

    # PPR summaries (only if paired-pulse data exists)
    ppr_image_summary = None
    ppr_group_summary = None
    if "ppr" in firings_per_roi.columns:
        ppr_image_summary = generate_ppr_image_summary(firings_per_roi)
        if ppr_image_summary is not None and len(ppr_image_summary) > 0:
            ppr_group_summary = generate_ppr_group_summary(ppr_image_summary)
            _msg(f"  PPR: {len(ppr_image_summary)} images with paired-pulse data")

    # ---- Save CSVs ----
    analysis_dir = os.path.join(output_folder, "csv_outputs", "analysis")
    csvs_dir = os.path.join(analysis_dir, "csvs")
    _safe_mkdir(csvs_dir)

    # Also save plain combined CSV
    combined_csv_dir = os.path.join(output_folder, "csv_outputs",
                                     "combined_csv")
    _safe_mkdir(combined_csv_dir)
    _save_csv(combined_df,
              os.path.join(combined_csv_dir,
                           "combined_SNRlabeled_traces.csv"))

    _save_csv(combined_with_peaks,
              os.path.join(csvs_dir, "combined_with_firings.csv"))
    _save_csv(firings_per_roi,
              os.path.join(csvs_dir, "firings_per_roi.csv"))
    _save_csv(per_image,
              os.path.join(csvs_dir, "firings_per_image.csv"))
    _save_csv(group_summary,
              os.path.join(csvs_dir, "group_summary.csv"))

    if ppr_image_summary is not None and len(ppr_image_summary) > 0:
        _save_csv(ppr_image_summary,
                  os.path.join(csvs_dir, "ppr_per_image.csv"))
    if ppr_group_summary is not None and len(ppr_group_summary) > 0:
        _save_csv(ppr_group_summary,
                  os.path.join(csvs_dir, "ppr_group_summary.csv"))

    # Save analysis parameters for reproducibility
    params_to_save = {k: v for k, v in params.items()}
    # Convert numpy types for JSON serialization
    for k, v in params_to_save.items():
        if isinstance(v, (np.integer,)):
            params_to_save[k] = int(v)
        elif isinstance(v, (np.floating,)):
            params_to_save[k] = float(v)
    with open(os.path.join(csvs_dir, "analysis_params.json"), "w") as f:
        json.dump(params_to_save, f, indent=2)

    _msg(f"  Saved CSVs to: {csvs_dir}")

    # ---- Step 4: plots ----
    _msg("Step 4/5: Generating plots ...")
    plots_dir = os.path.join(analysis_dir, "plots")
    generate_all_plots(combined_with_peaks, firings_per_roi, groups,
                       plots_dir, ppr_image_summary=ppr_image_summary,
                       per_image_summary=per_image)
    _msg(f"  Saved plots to: {plots_dir}")

    # ---- Step 5: auto-save config alongside results ----
    _msg("Step 5/5: Saving config ...")
    config_path = os.path.join(csvs_dir, "analysis_params.json")
    _msg(f"  Config saved to: {config_path}")

    _msg("Analysis complete!")

    return {
        "combined_with_peaks": combined_with_peaks,
        "firings_per_roi": firings_per_roi,
        "per_image": per_image,
        "group_summary": group_summary,
        "ppr_image_summary": ppr_image_summary,
        "ppr_group_summary": ppr_group_summary,
        "analysis_dir": analysis_dir,
    }


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python trace_analysis.py <output_folder> [frame_rate]")
        print("  Additional params can be set via environment variables:")
        print("    SD_MULTIPLIER, ROLLING_WINDOW, ORDER")
        print("    BASELINE_START_FRAME, BASELINE_END_FRAME")
        print("    STIM_START_FRAME, RESPONSE_WINDOW")
        print("    PPR_PULSE1_FRAME, PPR_PULSE2_FRAME, PPR_RESPONSE_WINDOW")
        print("    GROUPS (comma-separated)")
        sys.exit(1)

    folder = sys.argv[1]
    fr = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0

    p = {
        "sd_multiplier":        float(os.environ.get("SD_MULTIPLIER", "4.0")),
        "rolling_window":       int(os.environ.get("ROLLING_WINDOW", "10")),
        "baseline_start_frame": int(os.environ.get("BASELINE_START_FRAME", "0")),
        "baseline_end_frame":   int(os.environ.get("BASELINE_END_FRAME", "49")),
        "order":                int(os.environ.get("ORDER", "1")),
        "frame_rate":           fr,
        "stim_start_frame":     int(os.environ.get("STIM_START_FRAME", "50")),
        "response_window":      int(os.environ.get("RESPONSE_WINDOW", "5")),
        "ppr_pulse1_frame":     int(os.environ.get("PPR_PULSE1_FRAME", "50")),
        "ppr_pulse2_frame":     int(os.environ.get("PPR_PULSE2_FRAME", "60")),
        "ppr_response_window":  int(os.environ.get("PPR_RESPONSE_WINDOW", "4")),
        "groups": os.environ.get("GROUPS", "WT,APOE").split(","),
    }

    run_full_analysis(folder, p)
