# Technical Reference

## Architecture

The toolbox is structured as a set of Python modules with a napari-based GUI frontend. Two separate conda environments are used: one for the viewer (Python 3.10, napari) and one for GPU processing (Python 3.9, ilastik, CuPy).

```
┌─────────────────────────────────────────────┐
│  viewer.py  (napari GUI)                    │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
│  │Processing│ │View      │ │Analyze      │ │
│  │Tab       │ │Results   │ │Results      │ │
│  └────┬─────┘ └────┬─────┘ └──────┬──────┘ │
│       │             │              │        │
│       ▼             │              ▼        │
│  subprocess ──┐     │    analyze_results.py │
│               │     │    trace_analysis.py  │
│               │     │    outlier_explorer.py│
└───────────────┼─────┼──────────────────────-┘
                │     │
                ▼     │
┌───────────────────┐ │
│processing_utils.py│ │   (iglusnfr_processing env)
│  CuPy / cuCIM    │ │
│  ilastik         │ │
│  joblib parallel │ │
└──────────────────-┘ │
                      │
      config_manager.py  ← shared (hierarchical JSON config)
      config.py          ← re-exports for backward compat
```

## Module Descriptions

### `viewer.py` (2900+ lines)

Main napari application. Contains:

- **`Config`** class: UI settings pulled from `config_manager.DEFAULT_CONFIG`
- **`ProcessingWidget`**: Tab 1 — folder selection, parameter controls, subprocess launch
- **`ProcessingWorker`** (QThread): Builds and runs processing scripts as subprocesses
- **`DatasetLoaderWidget`**: Part of Tab 2 — folder scanning, dataset dropdown, file loading
- **`ROIPlotterWidget`**: Part of Tab 2 — Plotly trace plots, firing detection, outlier explorer button
- **`AnalyzeResultsWidget`**: Tab 3 — batch analysis configuration and execution
- **`NapariViewer`**: Main class — creates napari window, manages layers, connects widgets

Key design decisions:
- Processing runs in a **subprocess** (separate Python environment) to avoid GPU library conflicts with the viewer
- Plotly plots are displayed in a **QWebEngineView** — the viewer sets `document.title` from JS hover events and reads it back in Python for frame sync
- The outlier layer uses **in-place data updates** (`layer.data = ...`) instead of remove/add to avoid Qt segfaults during napari render cycles

### `processing_utils.py`

GPU-accelerated image processing pipeline:

1. **Background estimation**: Percentile filter on each z-slice (CuPy `percentile_filter`)
2. **Background subtraction**: GPU float32 subtraction (CuPy)
3. **Normalization**: Per-pixel division by background (CuPy)
4. **ilastik classification**: Pixel classification via ilastik headless API
5. **Connected component labeling**: cuCIM GPU-accelerated labeling
6. **Temporal color coding**: Assigns color by first-appearance timepoint
7. **Region measurement**: Parallel per-timepoint measurement (`joblib` with configurable `n_jobs` and `parallel_backend`)
8. **Area filtering**: Min/max area thresholds
9. **Hierarchical clustering**: Ward linkage on centroid distances
10. **IoU cluster merging**: Merges overlapping clusters
11. **SNR filtering**: Accepts/rejects clusters based on signal-to-noise ratio
12. **Morphology extraction**: `eccentricity`, `solidity`, `perimeter`, `circularity` via `skimage.regionprops_table`

### `trace_analysis.py`

Core analysis module:

- **`detect_firings()`**: Rolling-mean threshold + local maxima detection. Returns accepted frames, baseline stats (median, MAD, p5, p95, slope), and threshold arrays.
- **`analyze_combined_traces()`**: Processes all ROIs, builds `firings_per_roi` DataFrame with morphology metrics, baseline stats, and evoked response quantification.
- **`compute_outlier_scores()`**: Per-metric modified z-scores (MAD-based) with configurable per-metric enable/threshold. Adds `outlier_flag` and `outlier_reasons` columns.
- **`quantify_evoked_train()`**: Counts responses within window after each stimulus.
- **`quantify_ppr()`**: Paired-pulse ratio = peak2/peak1 intensity.
- **Summary generators**: `generate_per_image_summary()`, `generate_group_summary()`, `generate_ppr_summaries()`.
- **Plot generators**: Box plots, histograms, traces, PPR plots, baseline sanity check, outlier summary.

### `analyze_results.py`

CSV discovery and combination:

- **`discover_csv_files()`**: Finds `*_full_SNRlabeled_*.csv` files recursively.
- **`extract_metadata_from_filename()`**: Regex-based extraction of group, experiment type, AP, Hz, PPR from filenames.
- **`combine_csv_files()`**: Concatenates all CSVs with metadata columns. Uses Polars for speed with Pandas fallback. Handles type casting, unnamed column cleanup.

### `outlier_explorer.py`

Interactive QDialog with matplotlib histograms:

- One histogram subplot per enabled metric with `matplotlib.widgets.RangeSlider`
- Slider callbacks recolor bars (blue = normal, red = outside range) and update flagged count
- "Apply to Viewer" sends `set[int]` of flagged ROI labels back to napari via callback
- OR-based flagging: ROI is flagged if outside range on ANY enabled metric

### `config_manager.py`

Hierarchical config resolution:

- **`DEFAULT_CONFIG`**: Single source of truth for all parameters (processing, viewing, analysis, UI)
- **`resolve_config()`**: Merges default → root config → subfolder config → GUI overrides
- **`check_analysis_consistency()`**: Warns when analysis parameters differ across subfolders
- **`generate_template_config()`**: Writes a template with all defaults

## Data Flow

### Processing

```
Input .ome.tif
  → processing_utils.ProcessImages()
  → Output:
     ├── csv_outputs/per_image_csv/*_full_SNRlabeled_*.csv
     ├── csv_outputs/per_image_csv/*_evetsOnly_*.csv
     ├── Segmented/*_accepted_*.tif
     ├── Segmented/*_rejected_*.tif
     └── BKG_subtracted_normalized/*.tif
```

### Analysis

```
csv_outputs/per_image_csv/*.csv
  → analyze_results.combine_csv_files()
  → combined DataFrame (all images + metadata)
  → trace_analysis.run_full_analysis()
  → Output:
     └── csv_outputs/analysis/
         ├── csvs/  (firings_per_roi, per_image, group_summary, PPR)
         └── plots/ (spontaneous/, evoked/, paired_pulse/)
```

## Key Algorithms

### Firing Detection

```
threshold[i] = rolling_mean[i] + SD_multiplier * baseline_SD
```

Where `baseline_SD` is computed from the user-defined baseline window (default frames 0-49). Local maxima (via `scipy.signal.argrelextrema`) exceeding the threshold are accepted as firings.

### Outlier Scoring (Batch)

For each metric per image:
1. Compute median and MAD (Median Absolute Deviation)
2. Modified z-score = |value - median| / (MAD / 0.6745)
3. If MAD = 0, fall back to standard deviation
4. Flag ROI if z-score >= threshold for any enabled metric
5. `outlier_reasons` column lists which metrics triggered the flag

### Outlier Explorer (Interactive)

Direct value-range selection on histograms. No z-scores — the user visually selects the acceptable range for each metric. Flagging is OR-based across enabled metrics.

### Paired-Pulse Ratio

```
PPR = peak_intensity_after_pulse2 / peak_intensity_after_pulse1
```

Where each peak is the maximum intensity within `ppr_response_window` frames after the pulse. PPR > 1 = facilitation, PPR < 1 = depression.

## Config Parameter Reference

### processing

| Key | Default | Description |
|-----|---------|-------------|
| `bkg_percentile` | 10 | Background estimation percentile |
| `max_distance` | 6 | Max clustering distance (pixels) |
| `min_area` | 20 | Min ROI area (pixels) |
| `max_area` | 400 | Max ROI area (pixels) |
| `snr_threshold` | 3.0 | SNR cutoff |
| `iou_threshold` | 0.4 | IoU threshold for cluster merging |
| `snr_start_frame` | null | SNR calc start frame (null = first) |
| `snr_end_frame` | null | SNR calc end frame (null = last) |
| `n_jobs` | 15 | Parallel workers for measurement |
| `parallel_backend` | "threading" | Joblib backend |

### viewing

| Key | Default | Description |
|-----|---------|-------------|
| `frame_rate` | 100.0 | Acquisition frame rate (Hz) |
| `sd_multiplier` | 4.0 | Firing detection threshold multiplier |
| `rolling_window` | 10 | Rolling mean window |
| `baseline_start_frame` | 0 | Baseline start |
| `baseline_end_frame` | 49 | Baseline end |

### analysis

| Key | Default | Description |
|-----|---------|-------------|
| `frame_rate` | 100.0 | Acquisition frame rate (Hz) |
| `groups` | ["WT","APOE"] | Group names to detect in filenames |
| `sd_multiplier` | 4.0 | Firing detection threshold multiplier |
| `rolling_window` | 10 | Rolling mean window |
| `baseline_start_frame` | 0 | Baseline start |
| `baseline_end_frame` | 49 | Baseline end |
| `order` | 1 | argrelextrema order |
| `stim_start_frame` | 50 | First stimulus frame |
| `response_window` | 5 | Frames after stimulus to check |
| `ppr_pulse1_frame` | 50 | PPR first pulse frame |
| `ppr_pulse2_frame` | 60 | PPR second pulse frame |
| `ppr_response_window` | 4 | PPR peak search window |
| `outlier_config` | {...} | Per-metric outlier enable/threshold |

### ui

| Key | Default | Description |
|-----|---------|-------------|
| `font_size` | 11 | Base font size (px) |
| `font_size_large` | 13 | Group box title font |
| `font_size_title` | 14 | Section title font |
| `min_panel_width` | 350 | Minimum control panel width |
