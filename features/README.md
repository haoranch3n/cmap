# Features Module

This module owns feature-generation steps from segmented/combined cell data.

Current implementation runs natively here with code migrated from:
- `cell_boxing/crop_cells.py` -> `features/crop_cells.py`
- `cell_qc/extract_features.py` -> `features/extract_features.py`

`features/crop_cells.py` writes per-cell TIFFs under `cell_boxing/` (Z padded
around each label). Optional `--also-full-z` adds the same cells with full
stack depth under `cell_boxing_full_z/` (XY margins unchanged). From
`pipelines/run_analysis_pipeline.py`, pass **`--also-full-z` before
`--passthrough`** so postprocess scripts are not given that flag.

## DINOv2 deep embeddings (optional)

`features/dinov2/` adds a frozen DINOv2 ViT-B/14 (2.5D) encoder that produces
a 768-D vector per cell from `cell_boxing/cell_*.tif`. The CLI lives at
`features/extract_dinov2_embeddings.py` and writes
`cell_qc/dinov2_embeddings.npy` plus a matching `.csv`. See
[`features/dinov2/README.md`](dinov2/README.md) for defaults, command
examples, and extension points.
