# iGluSnFR Analysis Toolbox v3

A napari-based toolbox for processing, visualizing, and analyzing iGluSnFR glutamate imaging data. Supports GPU-accelerated image processing, interactive ROI inspection, automated firing detection, evoked/spontaneous/paired-pulse analysis, and interactive outlier exploration.

## Features

- **Processing Pipeline** — Background subtraction, ilastik pixel classification, connected-component labeling, hierarchical clustering, SNR filtering. GPU-accelerated via CuPy/cuCIM.
- **View Results** — Load processed datasets in napari, inspect ROI traces with interactive Plotly plots, search ROIs by ID, detect firings in real-time.
- **Analyze Results** — Combine per-image CSVs, extract metadata from filenames, detect firings across all ROIs, generate summary statistics and publication-ready plots.
- **Outlier Explorer** — Interactive histogram-based outlier detection with drag-to-select range sliders per metric.
- **Config System** — Hierarchical JSON config files (`iglusnfr_config.json`) for per-subfolder parameter customization with GUI override support.

## Quick Start

```bash
# 1. Clone / copy this folder to your machine
# 2. Run setup (creates conda environments)
chmod +x setup.sh
./setup.sh

# 3. Launch the viewer
./run_viewer.sh
```

## Documentation

| Document | Description |
|----------|-------------|
| [Setup Guide](docs/SETUP_GUIDE.md) | Installation, environment creation, GPU/driver requirements |
| [User Guide](docs/USER_GUIDE.md) | How to use each tab in the GUI |
| [Analysis Results](docs/ANALYSIS_RESULTS.md) | Detailed description of analysis outputs, plots, and CSV files |
| [Processing Steps](docs/PROCESSING_STEPS.md) | Pipeline steps, parameter effects, skip-segmentation behavior |
| [Technical Reference](docs/TECHNICAL_REFERENCE.md) | Code architecture, module descriptions, algorithm details |

## Requirements

- **OS**: Linux (tested on Ubuntu 22.04/24.04)
- **GPU**: NVIDIA GPU with CUDA 12.x support (driver >= 525; >= 570 for Blackwell)
- **Conda**: Miniconda or Anaconda
- **Display**: X11 or Wayland with Qt support (for napari GUI)

## File Structure

```
├── viewer.py                  # Main napari application (3 tabs)
├── processing_utils.py        # GPU-accelerated image processing pipeline
├── trace_analysis.py          # Firing detection, summaries, plots
├── analyze_results.py         # CSV discovery, metadata extraction, combination
├── outlier_explorer.py        # Interactive histogram outlier dialog
├── config_manager.py          # Hierarchical JSON config resolution
├── config.py                  # Legacy config re-exports and env helpers
├── test_pipeline.py           # CLI pipeline test script
├── setup.sh                   # Environment creation script
├── run_viewer.sh              # Viewer launcher
├── environment_viewer.yml     # Conda env spec for viewer
├── environment_processing.yml # Conda env spec for processing
└── docs/                      # Documentation
```

## Environments

The toolbox uses two separate conda environments:

| Environment | Python | Purpose |
|-------------|--------|---------|
| `iglusnfr_viewer` | 3.10 | napari GUI, plotting, analysis |
| `iglusnfr_processing` | 3.9 | ilastik, CuPy, GPU processing |

The viewer launches processing as a subprocess in the processing environment, so both must be installed.
