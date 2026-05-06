---
name: Modularize Segmentation Workspace
overview: Reorganize cmap into top-level modules where segmentation only handles raw image to masks, and postprocess/qc/napari-plugin are independent sibling modules for clean expansion.
todos:
  - id: define-module-boundaries
    content: Define strict module boundaries so segmentation ends at mask outputs and downstream modules consume masks.
    status: pending
  - id: create-top-level-layout
    content: Create top-level folders under cmap for segmentation, postprocess, features, qc, visualization, and napari-plugin integration.
    status: pending
  - id: remap-code
    content: Move existing scripts into their target modules and add compatibility wrappers for old paths.
    status: pending
  - id: shared-contracts
    content: Add shared config and IO contracts that define artifacts passed between modules.
    status: pending
  - id: archive-legacy
    content: Archive legacy segmentation variants and date-specific scripts into a clearly marked legacy area.
    status: pending
  - id: docs-flow
    content: Update docs with module responsibilities, interfaces, and end-to-end flowchart from image input to visualization.
    status: pending
isProject: false
---

# Modularize CMAP As Sibling Modules

## Goals

- Keep `segmentation/` focused: **raw image input -> segmentation masks output only**.
- Make `postprocess/`, `features/`, `qc/`, and `napari-plugin/` sibling modules under `cmap/`.
- Improve long-term extensibility so each step can evolve independently.
- Preserve current behavior via compatibility entrypoints during migration.

## Target Layout Under `cmap/`

- `cmap/segmentation/`
  - preprocess + 2D segmentation + 3D assembly
  - contract: input raw volumes, output 3D mask artifacts
- `cmap/postprocess/`
  - mask filtering/combination logic
  - contract: input segmentation masks (+ selected channels), output filtered mask products
- `cmap/features/`
  - cell boxing + per-cell feature extraction
  - contract: input filtered/combined volumes, output feature tables
- `cmap/qc/`
  - thresholding, pass/fail tagging, QC reports
  - contract: input feature tables, output filtered feature tables and QC decisions
- `cmap/visualization/`
  - t-SNE/UMAP plots and analysis renderers
- `cmap/napari-plugin/`
  - viewer and interactive exploration layer
- `cmap/pipelines/`
  - orchestrators that wire modules without owning module logic
- `cmap/shared/`
  - shared config, logging, IO contracts
- `cmap/scripts/`
  - SLURM/LSF wrappers only
- `cmap/legacy/`
  - old folder variants and one-off historical reruns

## Current Code Mapping (Old -> New Module)

- `segmentation_multiscale_cellpose_3D/preprocessing/*` -> `cmap/segmentation/preprocess/`
- `segmentation_multiscale_cellpose_3D/multiscale_cellpose/*` -> `cmap/segmentation/cellpose2d/`
- `segmentation_multiscale_cellpose_3D/cellcomposor/*` -> `cmap/segmentation/assemble3d/`
- `segmentation_multiscale_cellpose_3D/postprocessing/*` -> `cmap/postprocess/`
- `segmentation_multiscale_cellpose_3D/cell_boxing/*` -> `cmap/features/cell_boxing/`
- `segmentation_multiscale_cellpose_3D/cell_qc/extract_features.py` -> `cmap/features/extract_features/`
- `segmentation_multiscale_cellpose_3D/cell_qc/filter_by_intensity.py` -> `cmap/qc/filter_by_intensity/`
- `segmentation_multiscale_cellpose_3D/cell_qc/tsne_visualize*.py` -> `cmap/visualization/embedding/`

## Module Interface Contracts

- `segmentation`: consumes raw volumes; produces mask artifacts only.
- `postprocess`: consumes masks + aligned intensity channels; produces cleaned/combined volumes.
- `features`: consumes cleaned/combined volumes; produces feature CSV/parquet.
- `qc`: consumes feature tables; produces pass/fail labels and filtered tables.
- `visualization`/`napari-plugin`: consume outputs from upstream modules; no back-coupled writes into segmentation internals.

## Migration Strategy

1. Stand up top-level folders under `cmap/` with empty module CLIs.
2. Move code into target modules while preserving old paths as wrappers.
3. Move orchestration to `cmap/pipelines/` and scheduler wrappers to `cmap/scripts/`.
4. Add shared contracts in `cmap/shared/io_contracts.py` to lock artifact formats.
5. Route runtime artifacts to a dedicated runtime root.
6. Archive old sibling segmentation variants into `cmap/legacy/` after parity checks.

## Pipeline Flow (Your Vision: Input -> Visualization)

```mermaid
flowchart LR
rawImages[RawImageVolumes] --> segmentationModule[segmentationModule]
segmentationModule --> segmentationMasks[SegmentationMasks]

segmentationMasks --> postprocessModule[postprocessModule]
postprocessModule --> cleanedVolumes[CleanedCombinedVolumes]

cleanedVolumes --> featuresModule[featuresModule]
featuresModule --> featureTable[FeatureTable]

featureTable --> qcModule[qcModule]
qcModule --> qcTable[QcFilteredTable]

qcTable --> visualizationModule[visualizationModule]
qcTable --> napariPluginModule[napariPluginModule]
visualizationModule --> plotsReports[PlotsAndReports]
napariPluginModule --> interactiveReview[InteractiveReview]
```



