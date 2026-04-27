# Segmentation Module

This module owns **raw image -> segmentation mask** processing only.

Core segmentation pipeline now runs natively from this folder.
A frozen copy of the old tree lives under
`archive/legacy_modules/segmentation_multiscale_cellpose_3D/` for reference only.

## Entrypoint

- `python segmentation/run_segmentation.py`
- `python segmentation/run_pipeline.py`

## Contract

- Input: raw image volumes
- Output: 3D segmentation masks
