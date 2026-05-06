# Batch DINOv2 embedding extraction (LSF + Slurm)

This mirrors how you would batch **canonical** `features/extract_features.py`:
one scheduler job **per sample**, each job runs with `--data-rel <sample_name>`
so `PIPELINE_OUTPUT_DIR/<sample>/cell_boxing/cell_*.tif` is processed and
`cell_qc/dinov2_embeddings.{npy,csv}` is written for that sample only.

## 1. Prepare `samples.txt`

List one sample directory name per line (the folder name under your pipeline
output root, e.g. `CGNSample1_Position0_decon_dsr`). Lines starting with `#`
and blank lines are ignored.

```bash
export CMAP_ROOT=/path/to/cmap
export PIPELINE_OUTPUT_DIR="${PIPELINE_OUTPUT_DIR:-$CMAP_ROOT/output}"

# Auto-discover: any immediate child of OUTPUT that has cell_boxing/
bash "$CMAP_ROOT/scripts/batch_dinov2/discover_samples.sh" > samples.txt
# Or edit samples.txt by hand.
```

## 2. Environment (optional)

Copy and edit:

```bash
cp "$CMAP_ROOT/scripts/batch_dinov2/env.example.sh" "$CMAP_ROOT/scripts/batch_dinov2/env.sh"
# Set CONDA_SH, CONDA_ENV or VENV_ACTIVATE, module loads, etc.
```

If `env.sh` exists next to the submit scripts, it is sourced automatically.

## 3. Dry-run one sample (login node / interactive)

```bash
cd "$CMAP_ROOT"
export PIPELINE_OUTPUT_DIR=/path/to/your/output   # if not default cmap/output
bash scripts/batch_dinov2/run_one_sample.sh YOUR_SAMPLE_NAME
```

Same Python invocation as canonical extraction, swapping the script:

```bash
python features/extract_dinov2_embeddings.py --data-rel YOUR_SAMPLE_NAME --device cuda
```

Add `--force` to overwrite existing `dinov2_embeddings.*`. Add `--apply-mask`
if you want primary-mask zeroing.

## 4. LSF: one `bsub` per sample (portable)

```bash
cd "$CMAP_ROOT"
mkdir -p scripts/batch_dinov2/logs
bash scripts/batch_dinov2/submit_lsf.sh scripts/batch_dinov2/samples.txt
```

Resource requests are not portable. Either export `LSF_RESOURCES` as a
**single** string of `bsub` flags before calling, or edit the default inside
`submit_lsf.sh`. Example:

```bash
export LSF_RESOURCES='-q gpu -W 8:00 -n 8 -R "rusage[mem=65536]" -gpu "num=1:j_exclusive=yes"'
bash scripts/batch_dinov2/submit_lsf.sh scripts/batch_dinov2/samples.txt
```

## 5. LSF: single array job (optional)

If your site supports job arrays, use `array_job.lsf` after editing paths and
`#BSUB` lines, then:

```bash
cd "$CMAP_ROOT"
NUM=$(wc -l < samples.txt)
bsub -J "dinov2[1-$NUM]" < scripts/batch_dinov2/array_job.lsf
```

The template uses `LSB_JOBINDEX` to pick line `LSB_JOBINDEX` from `samples.txt`.

## 6. Slurm: array over `samples.txt`

Edit `scripts/batch_dinov2/slurm_array.sbatch` once for your site (`partition`,
`gres`, `time`, `account`). Then from the repo root:

```bash
cd "$CMAP_ROOT"
mkdir -p scripts/batch_dinov2/logs
bash scripts/batch_dinov2/discover_samples.sh > scripts/batch_dinov2/samples.txt
bash scripts/batch_dinov2/submit_slurm.sh
```

`submit_slurm.sh` runs `sbatch --array=1-N%THROTTLE` and passes
`SAMPLES_FILE` via `--export`; it does **not** rewrite the tracked `.sbatch`
file. Override throttle with `SLURM_ARRAY_THROTTLE` (default `20`).

## 7. Pool + UMAP after all per-sample jobs finish

Single light job (CPU + `umap-learn`):

```bash
cd "$CMAP_ROOT"
export PIPELINE_OUTPUT_DIR=/path/to/your/output
python visualization/dinov2_visualize.py \
  --filter-col pass_intensity \
  --suffix pxfiltered
```

On LSF you might `bsub` a follow-up with `-w 'ended(dinov2_array_jobname)'`.
On Slurm use `--dependency=afterok:<jobid>` on a second `sbatch`.

## Canonical feature extraction (for comparison)

Same batching idea, different script:

```bash
python features/extract_features.py --data-rel YOUR_SAMPLE_NAME
```
