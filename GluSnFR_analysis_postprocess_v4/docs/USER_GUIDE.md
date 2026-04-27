# User Guide

## Overview

The iGluSnFR Analysis Toolbox has three main tabs:

1. **Processing** — Run the image processing pipeline
2. **View Results** — Inspect processed datasets interactively
3. **Analyze Results** — Batch analysis with summary statistics and plots

## Tab 1: Processing

### Workflow (top to bottom)

| Step | Section | What to do |
|------|---------|------------|
| 1 | **Input Data** | Browse to the folder containing `.ome.tif` images. The tool scans recursively for all images. |
| 2 | **Config File** | Optionally generate a template `iglusnfr_config.json` in the input folder. Edit it to set per-subfolder parameters. Check "Override config" to use GUI values instead. |
| 3 | **Output Directory** | Browse to the folder where results will be saved. Directory structure mirrors the input. |
| 4 | **ilastik Model** | Browse to your trained `.ilp` pixel classification model. |
| 5 | **Processing Parameters** | Adjust background percentile, cluster distance, area range, and SNR threshold. These values are used unless a config file overrides them. |
| 6 | **Run** | Select the processing environment, optionally check "Skip segmentation" if re-processing, then click **Run Processing**. |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Background Percentile | 10 | Percentile filter for background estimation |
| Max Cluster Distance | 6 | Maximum distance (pixels) for hierarchical clustering |
| Min Area | 20 | Minimum ROI area in pixels |
| Max Area | 400 | Maximum ROI area in pixels |
| SNR Threshold | 3.0 | Signal-to-noise ratio cutoff for accepting ROI clusters |

### Skip Segmentation

When checked, steps 1-6 (background subtraction, ilastik classification) are skipped and previously saved intermediate files are loaded. Steps 7-17 (labeling, clustering, filtering) always re-run. Use this to quickly re-cluster with different parameters.

## Tab 2: View Results

### Loading a Dataset

1. In the left panel, browse to the **output folder** from processing
2. Select a dataset from the dropdown
3. Choose "Accepted ROIs" or "Rejected ROIs"
4. The napari viewer shows the image with color-coded ROI labels

### Inspecting ROIs

- **Shift+Click** on any ROI to see its time-series trace in the Plotly plot
- **"Go to ROI"** text box: type an ROI ID number and press Enter to jump to it
- **"Keep previous"** checkbox: overlay multiple ROI traces for comparison
- **Hover** over the plot to navigate frames in the napari viewer

### Firing Detection

Enable firing detection to see accepted peaks overlaid on traces:

| Parameter | Default | Description |
|-----------|---------|-------------|
| SD Multiplier | 4.0 | Threshold = rolling_mean + SD_mult * baseline_SD |
| Rolling Window | 10 | Window size for centred rolling mean |
| Baseline Start | 0 | First frame of baseline window |
| Baseline End | 49 | Last frame of baseline window |

### Outlier Explorer

Click **"Open Outlier Explorer"** to launch an interactive histogram dialog:

1. Checkboxes at the top enable/disable metrics (baseline median, baseline SD, max signal, firing count, area, solidity, circularity)
2. Each metric shows a histogram of values across all ROIs
3. Drag the **range slider** handles to define the acceptable range
4. Histogram bars outside the range turn red; a live counter shows how many ROIs are flagged
5. Click **"Apply to Viewer"** to color-code ROIs in napari (red = flagged, green = normal)

## Tab 3: Analyze Results

### Workflow (top to bottom)

| Step | Section | What to do |
|------|---------|------------|
| 1 | **Output Folder** | Browse to the root output folder containing processed CSVs. |
| 2 | **Config File** | Manage config files. Override toggle, save/load config, generate templates. Warnings appear if analysis parameters differ across subfolders. |
| 3 | **Detected Metadata** | Review auto-detected metadata: groups, experiment types, AP/Hz values. |
| 4 | **Groups** | Edit the comma-separated list of group names (e.g., "WT, APOE, KO"). Case-insensitive. |
| 5 | **Acquisition & Baseline** | Set frame rate and baseline window. |
| 6 | **Firing Detection** | Set SD multiplier, rolling window, and local max order. |
| 7 | **Evoked Experiment** | Set stimulus start frame and response window for train stimulation. |
| 8 | **Paired-Pulse (PPR)** | Set pulse frames and PPR response window for paired-pulse experiments. |
| 9 | **Outlier Detection** | Enable/disable per-metric z-score outlier flags with individual thresholds. |
| 10 | **Run** | Click **RUN FULL ANALYSIS**. |

### Metadata Extraction

Filenames are parsed (case-insensitive) for:

| Field | Detection | Example |
|-------|-----------|---------|
| Group | Matches group names from the Groups field | `DIV14WT_spon.ome.tif` → WT |
| Experiment type | `spon` → spontaneous; AP/Hz/PPR → evoked | `_10AP_20Hz_` → evoked |
| Action potentials | `(\d+)AP` with word-boundary guards | `_100AP_` → 100 |
| Stimulus frequency | `(\d+)Hz` | `_20Hz_` → 20 |
| PPR | `_PPR_` in filename | paired-pulse evoked |

### Outputs

All results are saved under `<output_folder>/csv_outputs/analysis/`. See [Analysis Results](ANALYSIS_RESULTS.md) for full details on CSV files and plots.

## Config File System

### How It Works

Place `iglusnfr_config.json` in any folder in your input hierarchy:

```
input_folder/
├── iglusnfr_config.json        ← root config (applies to all)
├── spontaneous/
│   ├── iglusnfr_config.json    ← overrides for spontaneous images
│   └── WT_spon_001.ome.tif
└── evoked/
    ├── iglusnfr_config.json    ← overrides for evoked images
    └── WT_100AP_20Hz.ome.tif
```

Resolution order (later wins):
1. Built-in defaults
2. Root `iglusnfr_config.json`
3. Subfolder `iglusnfr_config.json` (closer to the image wins)
4. GUI values (when "Override config" is checked)

### Generating a Template

Click **"Generate Template Config"** in the Processing or Analyze Results tab. This writes a file with all default values that you can edit.

### GUI Override Toggle

When checked, GUI parameter values take precedence over config files. Useful for quick experiments before committing changes to the JSON file.

## Tips

- **Small screens** (< 1080p height): The tool auto-adjusts layout. All tab content is scrollable.
- **ROI labels**: ROI ID numbers are displayed as text labels next to each ROI in the viewer for easy identification.
- **Batch processing**: The processing tab processes all images sequentially. Monitor progress in the log area.
- **Re-analysis**: You can re-run analysis with different parameters without re-processing. Just change settings in the Analyze Results tab and click Run again.
