# CMAP 3D Cell Segmentation Pipeline — Overview

This pipeline segments cell nuclei from volumetric (3D) microscopy data. It takes multi-page TIFF stacks as input and produces 3D labeled cell masks as output, with optional multi-channel quality filtering and per-cell feature extraction.

## Pipeline Steps

### Core Segmentation

`python run_pipeline.py` runs steps 1–4 end-to-end. Place input TIFs under `data/`.

1. **Preprocessing** (`preprocessing/separate_channels.py`)
   Splits each 3D `(Z, Y, X)` or 4D `(Z, C, Y, X)` TIFF volume into individual 2D Z-plane images (DAPI channel only).

2. **2D Segmentation** (`multiscale_cellpose/segmentation_cellpose_2d.py`)
   Runs Cellpose (`cpsam` model) at multiple diameters (50–100 px), then merges the multi-scale masks into one final mask per Z-plane. Small objects are filtered and over-merged cells are split.

3. **Z-Stack Assembly** (`cellcomposor/stack_2D_planes.py`)
   Re-stacks the per-slice 2D masks into a single 3D volume per sample.

4. **3D Label Matching** (`cellcomposor/create_3D_cells.py`)
   Matches cell labels across Z-slices using Jaccard Index overlap, then applies 3D post-processing:
   - Gap bridging across missing slices
   - Short-fragment absorption into longer cells
   - Disconnected-component splitting
   - Z-span filtering (min 5 slices)
   - Size-consistency trimming (max 3× consecutive-slice area change)
   - Volume filtering (min 100 voxels)

### Post-processing (per-sample)

Run individually with `--data-rel <relative_path>` (e.g. `--data-rel 4_18_25/CGNSample3_Position8_decon_dsr`).

5. **Multi-channel filtering** (`postprocessing/filter_642_mask.py`)
   For 3-channel data (642 / 488 / 560 nm): keeps only 642 nm nuclei that overlap with both 488 and 560 masks. Writes a filtered mask and a 4-channel combined TIF.

6. **Combine with original** (`postprocessing/combine_original_with_mask.py`)
   Merges the original intensity volume with the segmentation mask into a 2-channel TIF for visual inspection.

### Cell-Level Analysis (per-sample)

7. **Cell cropping** (`cell_boxing/crop_cells.py`)
   Extracts each labeled cell as a 5-channel 3D crop (642 / 488 / 560 intensity + binary mask + boundary) with a configurable margin. Detects neighboring cells. Writes `summary.csv`.

8. **QC feature extraction** (`cell_qc/extract_features.py`)
   Computes per-cell features from the cropped TIFs: mean / median / std / total intensity per channel, cross-channel Pearson correlations, and volume. Writes `qc_features.csv`.

9. **Intensity filtering** (`cell_qc/filter_by_intensity.py`)
   Applies a threshold (log-Otsu by default) to flag dim cells as pass/fail across all three channels. Writes `qc_features_filtered.csv`.

## How to Run

| Task | Command |
|------|---------|
| Full core pipeline | `python run_pipeline.py` |
| Per-sample post-processing | `python postprocessing/filter_642_mask.py --data-rel <REL>` |
| Per-sample cell cropping + QC | `python cell_boxing/crop_cells.py --data-rel <REL>` then `python cell_qc/extract_features.py --data-rel <REL>` then `python cell_qc/filter_by_intensity.py --data-rel <REL>` |
| Batch jobs (SLURM / LSF) | Shell scripts under `scripts/` |

## Configuration

All shared paths and parameters live in **`pipeline_config.py`**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CELLPOSE_PRETRAINED_MODEL` | `cpsam` | Cellpose 4 pretrained model |
| `CELLPOSE_DIAMETERS` | 50–100 (step 10) | Diameters for multiscale segmentation |
| `AREA_THRESHOLD` | π × 20² ≈ 1257 px | Min 2D cell area |
| `JI_THRESHOLD` | 0.1 | Jaccard Index threshold for 3D matching |
| `MIN_CELL_Z_SPAN` | 5 slices | Min z-extent to keep a cell |
| `MIN_CELL_VOLUME_3D` | 100 voxels | Min 3D volume to keep a cell |
| `MAX_AREA_CHANGE_RATIO` | 3.0 | Max consecutive-slice area fold-change |

Input/output directories default to `data/` and `output/` under the project root but can be overridden via environment variables `PIPELINE_DATA_DIR` and `PIPELINE_OUTPUT_DIR`.

## Requirements

Python 3, `cellpose>=4`, `tifffile`, `scikit-image`, `scipy`, `numpy`, `matplotlib`. See `requirements.txt` for the full list.
