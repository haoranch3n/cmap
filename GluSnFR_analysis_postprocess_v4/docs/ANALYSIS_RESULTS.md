# Analyze Results — Documentation

## Overview

The **Analyze Results** tab combines per-image CSV outputs from the processing
pipeline into a single dataset, detects synaptic firing events, quantifies
evoked responses, computes summary statistics, and generates plots. All results
are saved under `<output_folder>/csv_outputs/analysis/`.

---

## 1. Pipeline Steps

| Step | Description |
|------|-------------|
| **1. Combine CSVs** | Discovers all `*_full_SNRlabeled_*.csv` files, concatenates them, and extracts metadata (group, experiment type, AP count, Hz) from each filename. |
| **2. Firing Detection** | For every ROI in the combined data, detects accepted firing events using a rolling-mean threshold. |
| **3. Summary Tables** | Aggregates firing counts and peak intensities at the ROI, image, and group levels. |
| **4. Plots** | Generates publication-ready PNG figures organized by experiment type. |
| **5. Config Save** | Saves all analysis parameters to `analysis_params.json` for reproducibility. |

---

## 2. Metadata Extraction from Filenames

The filename of each image is parsed (case-insensitive) for:

| Field | Detection Rule | Examples |
|-------|---------------|----------|
| **Group** | Matches any user-supplied group name (default: `WT`, `APOE`) | `DIV14WT_...` → WT |
| **Experiment type** | `spon` → spontaneous; `evok`, AP, Hz, or PPR → evoked | `_spont_` → spontaneous |
| **Evoked subtype** | `_PPR_` → paired\_pulse; AP without Hz → single\_stim; AP+Hz → train | `_PPR_20Hz_` → paired\_pulse |
| **Action potentials** | `(\d+)AP` with word-boundary guards (avoids matching APOE) | `_100AP_` → 100 |
| **Stimulus frequency** | `(\d+)Hz` | `_20Hz_` → 20 |

For **paired-pulse** files, if no AP count is in the filename, it defaults to 2.

---

## 3. Firing Detection Algorithm

Applied to every ROI trace regardless of experiment type.

### Method

1. **Baseline estimation** — Compute mean and SD of the signal within a
   user-defined frame window (default: frames 0–49, the pre-stimulus period).
   Both bounds are clamped to the actual trace length.

2. **Rolling mean** — A centred rolling mean of the signal is computed with a
   configurable window size (default: 10 frames).

3. **Adaptive threshold** — At each frame:

   ```
   threshold[i] = rolling_mean[i] + SD_multiplier × baseline_SD
   ```

   Default SD multiplier: **4.0**.

4. **Local maxima** — `scipy.signal.argrelextrema` finds local maxima with a
   configurable `order` parameter (default: 1 = compare with immediate
   neighbours).

5. **Acceptance** — A local maximum is accepted as a firing if its intensity
   exceeds the threshold at that frame.

### Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| SD Multiplier | 4.0 | Scales baseline SD for threshold |
| Rolling Window | 10 | Centred rolling-mean window (frames) |
| Baseline Start Frame | 0 | First frame of baseline window (inclusive) |
| Baseline End Frame | 49 | Last frame of baseline window (inclusive) |
| Local Max Order | 1 | Points on each side for `argrelextrema` |

---

## 4. Quantification by Experiment Type

### 4.1 Spontaneous

For spontaneous recordings (no stimulation), only the general firing detection
(Section 3) is applied. The output includes:

- **Total firings** per ROI
- **Mean and max peak intensity** per ROI
- **Baseline SD and mean** per ROI

### 4.2 Evoked — Train

For train stimulation experiments (e.g. 100 AP at 20 Hz):

1. **Stimulus frame computation** — Given `n_ap`, `stim_hz`, `frame_rate`, and
   `stim_start_frame`:

   ```
   stim_interval = round(frame_rate / stim_hz)
   stim_frames = [stim_start + i × stim_interval  for i in 0..n_ap-1]
   ```

   Example: 100 AP at 20 Hz, frame rate 100 Hz, start frame 50 →
   interval = 5 frames, last stimulus at frame 545.

2. **Response quantification** — For each stimulus frame, the algorithm checks
   whether any accepted firing falls within a **response window** (default: 5
   frames after the stimulus). If so, that stimulus counts as having received a
   response.

3. **Response rate** = (stimuli with a response) / (total stimuli).

#### Additional outputs per ROI

| Column | Description |
|--------|-------------|
| `stim_responses` | Number of stimuli that received a response |
| `stim_total` | Total number of stimuli |
| `response_rate` | stim\_responses / stim\_total |

### 4.3 Evoked — Single Stimulus

Single-stimulus experiments (1 AP, no Hz specified) are treated identically to
train experiments with `n_ap = 1`. The stimulus frame is `stim_start_frame`
(default 50). Response rate is either 0 or 1.

### 4.4 Evoked — Paired-Pulse (PPR)

Paired-pulse experiments (`_PPR_` in filename) receive a specialized analysis:

1. **Two defined pulse frames** — Configurable via the UI (default: pulse 1 at
   frame 50, pulse 2 at frame 60 for 20 Hz PPR).

2. **Peak search per pulse** — For each ROI, the algorithm finds the **maximum
   intensity** within the PPR response window after each pulse:
   - Pulse 1 window: `[pulse1_frame + 1, pulse1_frame + ppr_response_window]`
   - Pulse 2 window: `[pulse2_frame + 1, pulse2_frame + ppr_response_window]`
   - Default PPR response window: **4 frames**.

3. **Paired-Pulse Ratio** — `PPR = peak2_intensity / peak1_intensity`.
   - PPR > 1 indicates **facilitation** (2nd response larger).
   - PPR < 1 indicates **depression** (2nd response smaller).

4. **Asynchronous events** — Local maxima that fall **outside** both response
   windows are counted as asynchronous peaks.

#### Additional outputs per ROI

| Column | Description |
|--------|-------------|
| `ppr_peak1_intensity` | Max intensity in window after 1st pulse |
| `ppr_peak1_frame` | Frame of that peak |
| `ppr_peak2_intensity` | Max intensity in window after 2nd pulse |
| `ppr_peak2_frame` | Frame of that peak |
| `ppr` | peak2 / peak1 ratio |
| `ppr_n_async` | Count of asynchronous (out-of-window) peaks |

#### PPR configurable parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| 1st Pulse Frame | 50 | Frame of the first stimulus |
| 2nd Pulse Frame | 60 | Frame of the second stimulus |
| PPR Response Window | 4 | Frames after each pulse to search for peak |

---

## 5. Summary Statistics

All group-level statistics use **per-image means as data points** (each image
contributes one value) rather than individual ROIs, preventing images with many
ROIs from dominating the statistics.

### 5.1 Per-ROI (`firings_per_roi.csv`)

One row per ROI across all images. Contains firing counts, peak intensities,
baseline stats, and (for evoked) response rate or PPR metrics.

### 5.2 Per-Image (`firings_per_image.csv`)

One row per image. Columns:

| Column | Description |
|--------|-------------|
| `n_rois` | Number of ROIs in the image |
| `total_firings` | Sum of firings across all ROIs |
| `mean_firings_per_roi` | Mean firings per ROI |
| `sd_firings_per_roi` | SD of firings across ROIs |
| `median_firings_per_roi` | Median firings per ROI |
| `mean_peak_intensity` | Mean peak intensity across ROIs |
| `max_peak_intensity` | Max peak intensity across ROIs |
| `mean_response_rate` | Mean evoked response rate (train only) |

### 5.3 Group Summary (`group_summary.csv`)

One row per group + experiment condition. Computed from per-image means:

| Column | Description |
|--------|-------------|
| `n_images` | Number of images in the group |
| `total_rois` | Total ROIs across all images |
| `mean_firings` | Mean of per-image mean firings |
| `sd_firings` | SD of per-image mean firings |
| `median_firings` | Median of per-image mean firings |
| `mean_peak_intensity` | Mean of per-image mean peak intensity |
| `mean_response_rate` | Mean of per-image response rates (train) |

### 5.4 PPR Per-Image (`ppr_per_image.csv`)

One row per paired-pulse image:

| Column | Description |
|--------|-------------|
| `n_rois` | ROIs analysed |
| `mean_peak1_intensity` | Mean intensity after 1st pulse (across ROIs) |
| `mean_peak2_intensity` | Mean intensity after 2nd pulse (across ROIs) |
| `mean_ppr` | Mean PPR across ROIs |
| `sd_ppr` / `median_ppr` | SD and median of per-ROI PPR |
| `ppr_from_means` | Ratio of mean\_peak2 / mean\_peak1 |
| `mean_async_events` | Mean asynchronous events per ROI |

### 5.5 PPR Group Summary (`ppr_group_summary.csv`)

One row per group. Computed from per-image PPR means:

| Column | Description |
|--------|-------------|
| `n_images` | Images in this group |
| `mean_peak1` / `sd_peak1` | Group stats for 1st-pulse response |
| `mean_peak2` / `sd_peak2` | Group stats for 2nd-pulse response |
| `mean_ppr` / `sd_ppr` / `median_ppr` | Group PPR statistics |

---

## 6. Output Directory Structure

```
<output_folder>/csv_outputs/
├── per_image_csv/              # Individual pipeline CSVs (one per image)
│   └── *.csv
├── combined_csv/
│   └── combined_SNRlabeled_traces.csv   # All images concatenated
└── analysis/
    ├── csvs/
    │   ├── combined_with_firings.csv    # Combined data + accepted_peak column
    │   ├── firings_per_roi.csv          # One row per ROI
    │   ├── firings_per_image.csv        # One row per image
    │   ├── group_summary.csv            # One row per group+condition
    │   ├── ppr_per_image.csv            # PPR stats per image (if PPR data)
    │   ├── ppr_group_summary.csv        # PPR stats per group (if PPR data)
    │   └── analysis_params.json         # All parameters used
    └── plots/
        ├── spontaneous/
        │   ├── traces_<GROUP>.png
        │   ├── traces_all_groups.png
        │   ├── firings_boxplot_by_group.png
        │   ├── peak_intensity_histogram.png
        │   └── peak_intensity_boxplot_by_group.png
        ├── evoked/
        │   └── <N>AP_<F>Hz/
        │       ├── firings_boxplot_by_group.png
        │       ├── response_rate_boxplot_by_group.png
        │       ├── peak_intensity_histogram.png
        │       └── peak_intensity_boxplot_by_group.png
        └── paired_pulse/
            ├── ppr_boxplot_by_group.png          # All PPR frequencies
            └── PPR_<F>Hz/
                └── ppr_boxplot_by_group.png      # Per-frequency PPR
```

---

## 7. Plots

### 7.1 Spontaneous Plots

| Plot | Description |
|------|-------------|
| **traces\_\<GROUP\>.png** | All individual ROI traces overlaid for one group, with mean ± SEM. Opacity auto-scales with ROI count. |
| **traces\_all\_groups.png** | Raw individual traces for all groups on one plot (no averaging), color-coded by group. |
| **firings\_boxplot\_by\_group.png** | Boxplot of mean firings per ROI, using **per-image means** as data points. Each dot = one image. |
| **peak\_intensity\_histogram.png** | Side-by-side histograms of accepted-peak intensities (one subplot per group, independent y-axes for easy comparison). Shared x-axis bins. |
| **peak\_intensity\_boxplot\_by\_group.png** | Boxplot of mean accepted-peak intensity, using **per-image means** as data points. |

### 7.2 Evoked Plots (per AP/Hz condition)

Generated in a subfolder for each unique AP + Hz combination (e.g. `100AP_20Hz/`):

| Plot | Description |
|------|-------------|
| **firings\_boxplot\_by\_group.png** | Same as spontaneous but for this evoked condition. |
| **response\_rate\_boxplot\_by\_group.png** | Boxplot of mean stimulus response rate per image. Y-axis 0–1. |
| **peak\_intensity\_histogram.png** | Side-by-side histograms of peak intensities. |
| **peak\_intensity\_boxplot\_by\_group.png** | Boxplot of per-image mean peak intensity. |

### 7.3 Paired-Pulse Plots

Generated under `paired_pulse/` (overall and per-Hz subfolders):

| Plot | Description |
|------|-------------|
| **ppr\_boxplot\_by\_group.png** | Three-panel figure: (1) PPR ratio by group with reference line at 1.0, (2) 1st-pulse response intensity, (3) 2nd-pulse response intensity. Each dot = one image mean. |

---

## 8. Config File

All analysis parameters are saved to `analysis_params.json` alongside the
results. This file is automatically reloaded when you re-open the same output
folder in the Analyze Results tab, restoring all spinner values.

You can also manually **Save Config** / **Load Config** via buttons in the UI
to share parameter sets across experiments.

### Parameters stored

```json
{
  "sd_multiplier": 4.0,
  "rolling_window": 10,
  "baseline_start_frame": 0,
  "baseline_end_frame": 49,
  "order": 1,
  "frame_rate": 100.0,
  "stim_start_frame": 50,
  "response_window": 5,
  "ppr_pulse1_frame": 50,
  "ppr_pulse2_frame": 60,
  "ppr_response_window": 4,
  "groups": ["WT", "APOE"]
}
```

---

## 9. Baseline Sanity Check

A 3-panel baseline sanity check plot (`baseline_sanity_check.png`) is generated
for every analysis run, regardless of experiment type:

| Panel | Metric | Purpose |
|-------|--------|---------|
| 1 | Baseline Median | Detects inter-image shifts in baseline fluorescence |
| 2 | Baseline MAD | Detects images with unusually noisy baselines |
| 3 | Baseline Slope | Detects linear drift (photobleaching, focus drift) |

Each dot represents one image's mean across ROIs. The plot is a quick visual
check that baselines are comparable across images and groups.

Per-ROI baseline statistics are included in `firings_per_roi.csv`:
`baseline_median`, `baseline_mad`, `baseline_p5`, `baseline_p95`, `baseline_slope`.
Aggregated versions appear in per-image and group summary CSVs.

---

## 10. Outlier Detection

### Batch mode (Analyze Results tab)

The `firings_per_roi.csv` includes per-metric modified z-scores (MAD-based)
and an `outlier_flag` column. Seven metrics are supported, each with an
independent enable toggle and z-score threshold:

| Metric | Default enabled | Default threshold |
|--------|:-:|:-:|
| `baseline_median` | Yes | 3.0 |
| `baseline_sd` | Yes | 3.0 |
| `max_signal` | No | 3.0 |
| `total_firings` | Yes | 3.0 |
| `area` | Yes | 3.0 |
| `solidity` | Yes | 3.0 |
| `circularity` | Yes | 3.0 |

An `outlier_reasons` column lists the specific metrics that triggered the flag.

Summary CSVs include `n_flagged_rois` and `pct_flagged_rois` columns.
An `outlier_summary.png` plot shows % flagged per image and max z-score
distributions.

### Interactive mode (View Results tab — Outlier Explorer)

The **Outlier Explorer** dialog provides histogram-based outlier selection
without z-scores. Users drag range sliders on per-metric histograms to define
acceptable value ranges. ROIs outside any enabled range are flagged and colored
red in the napari viewer.

---

## 11. Statistical Design Notes

- **Boxplots and group summaries** use **per-image means** as data points
  (N = number of images), not individual ROIs. This prevents images with many
  ROIs from inflating sample size.
- **Histograms** use individual peak values (not averaged) but are shown in
  side-by-side subplots with independent y-axes so groups with different firing
  counts remain visually comparable.
- **PPR** is computed per-ROI, then averaged per-image. Group-level stats are
  derived from the per-image means.
