# Copilot Instructions — CMAP

## Build / install

The napari plugin (`napari-plugin/`) is the only installable Python package in
this repo. All other modules are run as scripts from the repo root.

```bash
# Install the plugin (editable)
cd napari-plugin
pip install -e .                   # minimal
pip install -e ".[all]"            # with Leiden, Louvain, UMAP

# DINOv2 extras (GPU work only; install on a compute node)
pip install -r features/dinov2/requirements.txt
```

## Tests

Tests live under `napari-plugin/tests/`, `features/`, `postprocess/tests/`, and
`features/dinov2/tests/`. Run them from the repo root:

```bash
# All napari-plugin tests
python -m pytest napari-plugin/tests/ -v

# Single test file
python -m pytest napari-plugin/tests/test_io.py -v

# Single test by name
python -m pytest napari-plugin/tests/test_embedding.py -v -k "test_leiden"

# Postprocess tests
python -m pytest postprocess/tests/ -v

# DINOv2 tests (requires torch)
python -m pytest features/dinov2/tests/ -v
```

## Architecture

The pipeline is a sequential chain of **sibling modules** under `cmap/`:

```
Raw images → segmentation/ → postprocess/ → features/ → qc/ → visualization/
                                                                    ↕
                                                             napari-plugin/
```

- **`segmentation/`** — raw volumes → 3D masks via multiscale Cellpose 2D + assembly.
  Entrypoint: `python segmentation/run_pipeline.py` (runs 4 sub-steps in order).
- **`postprocess/`** — mask filtering (`filter_642_mask.py`) and merging
  (`combine_with_mask.py`); also supports a `union_488_560` variant.
- **`features/`** — cell crops (`crop_cells.py`) and per-cell feature tables
  (`extract_features.py`). Optional DINOv2 768-D embeddings in `features/dinov2/`.
- **`qc/`** — intensity-based filters, merges QC CSVs across samples.
- **`visualization/`** — t-SNE / UMAP plots; `dinov2_visualize.py` pools DINOv2
  embeddings into `cell_qc_all/`.
- **`napari-plugin/`** — standalone installable QC annotation + embedding explorer.
  Package name: `cell_qc_plugin` (v0.2.2). Source under `napari-plugin/src/`.
- **`pipelines/run_analysis_pipeline.py`** — end-to-end orchestrator; pass
  `--run-dinov2` / `--run-dinov2-vis` to include the embedding steps.
- **`shared/`** — thin cross-module utilities (no heavy imports).
- **`archive/legacy_modules/`** — reference-only copies of old trees; **not part
  of the active pipeline**.

## Environment variables

| Variable | Module | Effect |
|---|---|---|
| `PIPELINE_OUTPUT_DIR` | `segmentation/config.py`, napari `scripts/_paths.py` | Override default `<repo>/output/` artifact root |
| `PIPELINE_DATA_DIR` | `segmentation/config.py` | Override default `<repo>/data/` input root |
| `CMAP_OUTPUT_DIR` | napari `scripts/_paths.py` | Fallback when `PIPELINE_OUTPUT_DIR` is unset |
| `CMAP_QC_ALL_CSV` | napari `scripts/_paths.py` | Override master embedding CSV path |
| `CMAP_REGISTERED_CELL_VIEW_ROOT` | napari `_io.py` | Optional root for affine-registered cell TIFFs |

## Key conventions

### Output paths — never hardcode legacy locations
All runtime artifacts go under `output/` (or `$PIPELINE_OUTPUT_DIR`).
**Never** use paths containing `segmentation_multiscale_cellpose_3D/output`.
The napari plugin and its helper scripts must point at the same root.

### Import pattern in module scripts
Scripts support being run from their own module directory *or* from the repo
root. The typical guard is:

```python
try:
    from config import ...          # run from module dir (e.g. cd segmentation)
except ModuleNotFoundError:
    from segmentation.config import ...  # run from repo root
```

### On-disk output layout
```
output/
  <experiment>/<sample>/
    cell_boxing/        # TIFF crops from features/crop_cells.py
    cell_boxing_full_z/ # (optional) full-Z crops with --also-full-z
    cell_qc/
      qc_features.csv
      tsne_embedding.csv   # features + pre-computed t-SNE / UMAP
      dinov2_embeddings.csv
  cell_qc_all/          # pooled cross-sample master CSVs
    tsne_embedding_all.csv
    dinov2_embedding_all.csv
```

### CSV column conventions (features)
- `cell_id` maps to filename `cell_{cell_id:04d}.tif` (zero-padded to 4 digits).
- Pre-computed embedding columns: `tsne_1_vol`, `tsne_2_vol`, `tsne_1_novol`,
  `tsne_2_novol`, `umap_1_vol`, `umap_2_vol`, `umap_1_novol`, `umap_2_novol`,
  `umap_dinov2_1`, `umap_dinov2_2`.
- Leiden clustering is done on **standardised feature space**, not on the 2-D
  embedding coordinates.

### HPC — never run heavy work on the login node
Submit via `bsub`:
- CPU-only: `#BSUB -q standard`
- GPU / Cellpose: `#BSUB -q rhel88_gpu`
- Debug one-slice Cellpose: `segmentation_multiscale_cellpose_3D/scripts/bsub_debug_cellpose_one_slice.sh`

**All LSF job scripts must use absolute paths** (`$PROJECT_ROOT/...`) because
LSF runs from a spool directory, not the submission cwd. Do not use relative
paths in bsub scripts.

Safe on the login node: `bsub`, `bjobs`, `bpeek`, log file inspection, `git`,
small read-only Python probes.

### Napari plugin: headless-safe imports
Heavy GUI imports (`napari`, `plotly`) are lazy in `cell_qc_plugin/__init__.py`.
Helper scripts can use:
```bash
PYTHONPATH=napari-plugin/src python napari-plugin/scripts/compute_umap_rgb_artifacts.py
```
without a full napari install.

### Git on this machine
If `git commit` fails with unknown `trailer` options, use a minimal environment:
```bash
env -i PATH=/usr/bin:/bin HOME="$HOME" git commit -m "..."
```

## Running the full pipeline

```bash
# End-to-end (segmentation → features → qc → visualization)
python pipelines/run_analysis_pipeline.py --help

# With DINOv2 embeddings + pooled UMAP
python pipelines/run_analysis_pipeline.py --run-dinov2-vis --passthrough --data-rel <rel>

# Segmentation only
python segmentation/run_pipeline.py

# Napari QC plugin
cd napari-plugin && ./run.sh [/path/to/output/root]
```
