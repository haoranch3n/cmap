# DAPI 2D / 3D segmentation pipeline

**Version:** 1.1

Workflow: **preprocess volumetric DAPI → run Cellpose 2D per Z plane → (optional) stack and track in 3D.**

This repo targets **DAPI-only** data. The older dual-channel (DAPI + cytoplasm) path, multiscale diameter merging, and mask postprocessing have been removed.

## Requirements

- Python 3 with `cellpose>=4` (see `requirements.txt`). Cellpose 4 uses the **`cpsam`** checkpoint (Cellpose-SAM) as the default 2D pretrained model.

## Paths

`pipeline_config.py` defines:

| Variable | Role |
|----------|------|
| `DATA_DIR` | Input volumes (e.g. `data/`) |
| `TIF_PLANES_DIR` | Per-slice `*_DAPI.tif` (`output/tif_planes/`) |
| `SEGMENTATION_2D_DIR` | One `*_final_mask.tif` per slice (`output/segmentation_2D_planes/`) |
| `SEGMENTATION_2D_STACKED_DIR` | Z-stacks of 2D masks (`output/segmentation_2D_stacked/`) |
| `SEGMENTATION_3D_DIR` | 3D label volumes (`output/segmentation_3D_masks/`) |

## Execution order

**All steps in one command** (from the project root):

```bash
python run_pipeline.py
```

Or run steps manually:

1. **Preprocessing** — split each TIFF volume into 2D Z planes (DAPI only):

   ```bash
   python preprocessing/separate_channels.py
   ```

   - Expects `DATA_DIR` to contain `*.tif` (or set `IMAGE_DIR` / `IMAGE_TYPE` inside `preprocessing/separate_channels.py`).
   - **3D** `(Z, Y, X)` and **4D** `(Z, C, Y, X)` are supported; only channel 0 is written for 4D stacks.

2. **2D segmentation** — Cellpose `cpsam`, one mask per plane:

   ```bash
   python multiscale_cellpose/segmentation_cellpose_2d.py
   ```

   Writes `<plane_stem>/<plane_stem>_final_mask.tif` under `SEGMENTATION_2D_DIR`.

3. **Z-stack 2D masks** — concatenate per-slice masks along Z:

   ```bash
   python cellcomposor/stack_2D_planes.py
   ```

   Supports the original nested folder layout (animal / section / slice) and a **flat** layout `segmentation_2D_planes/<volume>/<slice_folder>/…` (used for single volumes under `data/`).

4. **3D label volume** — match labels between slices and write indexed + RGB preview:

   ```bash
   python cellcomposor/create_3D_cells.py
   ```

   Main output: `output/segmentation_3D_masks/<volume>/<volume_stem>_3D_indexed.tif` (3D `uint16` labels), plus `<volume_stem>_3D_color.tif` for RGB preview.

## Test data

Place volumes under `data/` (for example `data/642nm_crop.tif`). Run step 1, then step 2.

## Notes

- Intensity quantification under `postprocessing/` is legacy; it is not part of the default workflow. Adjust paths there if you still need it.
- Custom `cellpose` checkpoints can be passed by editing `CELLPOSE_PRETRAINED_MODEL` in `pipeline_config.py` or the arguments to `run_segmentation()` in `segmentation_cellpose_2d.py`.
