# Migration Map (Copy-First)

This migration follows:
1. copy original script behavior into new module path,
2. apply minimal path/import edits only,
3. remove dead compatibility shims once native modules are verified.

## Segmentation (native under `cmap/segmentation`)

- `segmentation_multiscale_cellpose_3D/pipeline_config.py` -> `segmentation/config.py`
- `segmentation_multiscale_cellpose_3D/preprocessing/separate_channels.py` -> `segmentation/preprocess/separate_channels.py`
- `segmentation_multiscale_cellpose_3D/multiscale_cellpose/segmentation_cellpose_2d.py` -> `segmentation/cellpose2d/segmentation_cellpose_2d.py`
- `segmentation_multiscale_cellpose_3D/cellcomposor/stack_2D_planes.py` -> `segmentation/assemble3d/stack_2d_planes.py`
- `segmentation_multiscale_cellpose_3D/cellcomposor/create_3D_cells.py` -> `segmentation/assemble3d/create_3d_cells.py`
- `segmentation_multiscale_cellpose_3D/cellcomposor/segmentation_3D/match_2D_cells.py` -> `segmentation/assemble3d/segmentation_3d/match_2d_cells.py`
- `segmentation_multiscale_cellpose_3D/run_pipeline.py` -> `segmentation/run_pipeline.py`

## Postprocess (native under `cmap/postprocess`)

- `segmentation_multiscale_cellpose_3D/postprocessing/filter_642_mask.py` -> `postprocess/filter_642_mask.py`
- `segmentation_multiscale_cellpose_3D/postprocessing/combine_original_with_mask.py` -> `postprocess/combine_with_mask.py`

## Features + QC (native under `cmap/features` and `cmap/qc`)

- `segmentation_multiscale_cellpose_3D/cell_boxing/crop_cells.py` -> `features/crop_cells.py`
- `segmentation_multiscale_cellpose_3D/cell_qc/extract_features.py` -> `features/extract_features.py`
- `segmentation_multiscale_cellpose_3D/cell_qc/filter_by_intensity.py` -> `qc/filter_by_intensity.py`

## Visualization (native under `cmap/visualization`)

- `segmentation_multiscale_cellpose_3D/cell_qc/tsne_visualize.py` -> `visualization/tsne_visualize.py`
- `segmentation_multiscale_cellpose_3D/cell_qc/tsne_visualize_all.py` -> `visualization/tsne_visualize_all.py`

## Archived Legacy Folders

Moved from top-level `cmap/` into `archive/legacy_modules/`:

- `segmentation_AICS/`
- `segmentation_cellpose_3D/`
- `segmentation_multiscale_cellpose_3D/`
- `segmentation_stardist/`

## Minimal Edit Rules Used

- `from pipeline_config import ...` -> `from config import ...`
- internal package path updates for moved modules
- top-level path bootstrap for execution from new module root
- entrypoint filenames updated to match new module tree

No algorithm-level changes are intended in this phase.

