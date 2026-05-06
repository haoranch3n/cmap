# Batch full-Z cell cropping (LSF + Slurm)

One scheduler job per sample. Each job runs
`features/crop_cells.py --also-full-z --data-dir <dir> --output-dir <dir>`
to produce `cell_boxing/` (padded Z) and `cell_boxing_full_z/` (full Z depth,
same XY margin) for that sample.

## 1. Prepare environment (optional)

```bash
cp scripts/batch_full_z_crop/env.example.sh scripts/batch_full_z_crop/env.sh
# Edit env.sh: set CONDA_SH + CONDA_ENV, or VENV_ACTIVATE, and PIPELINE_OUTPUT_DIR
```

If `env.sh` exists next to the submit scripts it is sourced automatically.

## 2. Dry-run one sample

```bash
cd "$CMAP_ROOT"
bash scripts/batch_full_z_crop/run_one_sample.sh /path/to/output/SampleX
```

## 3. LSF: one `bsub` per sample

```bash
cd "$CMAP_ROOT"
bash scripts/batch_full_z_crop/submit_lsf.sh            # uses PIPELINE_OUTPUT_DIR
bash scripts/batch_full_z_crop/submit_lsf.sh /path/to/output  # explicit root
```

The script auto-discovers samples via `batch_crop_cells_full_z.py --write-list`,
then submits one `bsub` per sample to the **standard** queue (no email).

Override resources:

```bash
export LSF_RESOURCES='-q standard -n 8 -R "rusage[mem=64000] span[hosts=1]" -W 24:00'
bash scripts/batch_full_z_crop/submit_lsf.sh
```

Pass `--force` to overwrite existing crops:

```bash
CROP_FORCE=1 bash scripts/batch_full_z_crop/submit_lsf.sh
```

## 4. LSF: single array job (optional)

```bash
cd "$CMAP_ROOT"
python scripts/batch_crop_cells_full_z.py \
  --output-root /path/to/output \
  --write-list logs/full_z_crop_paths.txt

N=$(wc -l < logs/full_z_crop_paths.txt)
bsub -J "fzcrop[1-$N]" \
  -env "all,PATHS_FILE=$PWD/logs/full_z_crop_paths.txt" \
  < scripts/batch_full_z_crop/array_job.lsf
```

## 5. Slurm: array over sample list

```bash
cd "$CMAP_ROOT"
bash scripts/batch_full_z_crop/submit_slurm.sh           # uses default output root
bash scripts/batch_full_z_crop/submit_slurm.sh /path/to/output
```

## Files

| File | Purpose |
|------|---------|
| `submit_lsf.sh` | One `bsub` per sample (auto-discovers samples) |
| `array_job.lsf` | LSF array job template |
| `submit_slurm.sh` | Slurm array submission |
| `slurm_array.sbatch` | Slurm array job template |
| `run_one_sample.sh` | Runs `crop_cells.py --also-full-z` for one sample |
| `env.example.sh` | Template for `env.sh` (Python env, paths, margins) |
