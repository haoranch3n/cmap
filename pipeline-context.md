# Pipeline context (CMAP)

Single reference for paths, environment variables, and how stages connect. Keep this in sync when you add stages or rename folders.

## Layout (under `cmap/`)

| Path | Role |
|------|------|
| `segmentation/` | Raw volumes → 2D/3D masks (`run_pipeline.py`, `config.py`) |
| `postprocess/` | Mask cleanup / merged products consumed by downstream steps |
| `features/` | Cell crops and per-cell feature tables |
| `qc/` | Feature filters, `qc_features_filtered.csv`, pooled tables under `output/cell_qc_all/` |
| `visualization/` | Embeddings and plots (e.g. t-SNE / UMAP CSVs) |
| `napari-plugin/` | Interactive QC; pass the **same** `output/` root the pipeline wrote |
| `pipelines/` | Orchestrators (e.g. `run_analysis_pipeline.py`) |
| `shared/` | Small cross-module helpers |
| `output/` | Default runtime artifact root (ignored by git) |
| `archive/legacy_modules/` | Old monolithic trees; not part of the active path |

## Environment variables

| Variable | Used by | Meaning |
|----------|---------|---------|
| `PIPELINE_OUTPUT_DIR` | `segmentation/config.py`, napari `scripts/_paths.py` | Absolute path replacing default `<cmap>/output` |
| `PIPELINE_DATA_DIR` | `segmentation/config.py` | Input data root (default `<cmap>/data`) |
| `CMAP_OUTPUT_DIR` | napari `scripts/_paths.py` | Fallback if `PIPELINE_OUTPUT_DIR` is unset (same semantic) |
| `CMAP_QC_ALL_CSV` | napari `scripts/_paths.py` | Override path to master embedding/features CSV (default `…/output/cell_qc_all/tsne_embedding_all.csv`) |
| `CMAP_REGISTERED_CELL_VIEW_ROOT` | napari `_io.py` | Optional root for affine-registered cell TIFFs (display only). When set, the viewer loads `{root}/{dataset}/{sample}/cell_NNNN/cell_NNNN_488_560_registered.tif` instead of the pipeline crop. Falls back to the pipeline TIFF when the registered file is absent. |

## On-disk conventions (typical)

Under the chosen output root:

- Per-experiment / per-sample trees with `cell_boxing/` (TIFFs) and `cell_qc/` (`qc_features.csv`, `qc_features_filtered.csv`).
- Pooled QC: `cell_qc_all/` (e.g. `tsne_embedding_all.csv`, filtered embedding exports).

The napari plugin discovers samples by walking this tree; symlinks under `napari-plugin/data/` are optional conveniences.

## Orchestration

- **`python pipelines/run_analysis_pipeline.py`** — end-to-end native chain; optional visualization flag as implemented in that script.
- Individual modules expose their own CLIs; prefer calling those from the orchestrator or documented order in `MODULE_ARCHITECTURE.md`.

## Legacy

Paths containing `segmentation_multiscale_cellpose_3D/output` are obsolete for **new** runs. Legacy copies may exist only under `archive/legacy_modules/` for comparison or one-off reruns.
