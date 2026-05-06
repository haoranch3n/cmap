# Agent guide (CMAP workspace)

Use this file at the start of a new chat so work stays aligned with the modular layout and paths on disk.

## Read first

1. **`pipeline-context.md`** — output directories, env vars, orchestrator entrypoints, napari data root, and where legacy code lives.
2. **`MODULE_ARCHITECTURE.md`** — module responsibilities and the high-level flowchart.
3. **`MIGRATION_MAP.md`** — where old script names moved after the copy-first migration.

## HPC (login vs LSF)

- **Login node:** avoid Cellpose, GPU work, full pipelines, and heavy TIFF/array processing. Use **`bsub`** (`standard` for CPU-only, **`rhel88_gpu`** for GPU). One-slice Cellpose probe: `segmentation_multiscale_cellpose_3D/scripts/bsub_debug_cellpose_one_slice.sh` (CPU) or `bsub_debug_cellpose_one_slice_gpu.sh` (GPU). See **`.cursor/rules/lsf-no-login-compute.mdc`**. Job scripts must use **absolute paths** (`$PROJECT_ROOT/...`) because LSF may run from a spool cwd.

## Quick execution

- Full analysis chain (segmentation → postprocess → features → qc → visualization): run from repo root, e.g. `python pipelines/run_analysis_pipeline.py --help` and pass the same flags your site uses for data/output.
- Segmentation-only: `python segmentation/run_pipeline.py` (see that module’s CLI).
- **Napari QC plugin**: install `napari-plugin/` in editable mode, then point the viewer at **`output/`** (the pipeline output root), not any path under `archive/`.

## Repo facts

- **`archive/legacy_modules/`** holds old trees (`segmentation_multiscale_cellpose_3D`, etc.); they are reference-only and typically **gitignored**.
- Default on-disk output is **`output/`** at the repo root unless overridden by **`PIPELINE_OUTPUT_DIR`** (see `segmentation/config.py`).

## Git on this machine

If `git commit` fails with unknown `trailer` options, a broken `GIT_*` or hook env may be involved; a minimal environment (`env -i PATH=/usr/bin:/bin HOME="$HOME" ...`) often succeeds.
