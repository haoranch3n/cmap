---
title: Per-Channel Intensity QC Filter — Background-Statistics Redesign
todos:
  - id: git-snapshot
    content: Commit current state on full-z-range-dev (skip logs/, cufile.log) and create bg-sigma-filter-dev branch off it
  - id: explore-script
    content: Write qc/bg_threshold_explore.py (post-clip) — 5 candidate methods (bg_sigma, bg_mad, otsu_or_bg, bg_frac, bg_pct95) plus baselines (log_otsu, otsu2). Eroded background, top-1% clip; sensitivity row with clipping disabled. Tracks union_488_560 cell 25.
  - id: explore-lsf
    content: Write qc/bsub_bg_threshold_explore.sh (standard queue, 16 GB) and submit on Sample7_Position7 for both variants; monitor and summarise.
  - id: pick-winner
    content: Review report with user; agree on winning method (must drop union_488_560 cell 25 and keep filtered_642 cell 25 / strong cells).
  - id: productionize
    content: Add chosen method(s) to qc/filter_by_intensity.py with --clip-negatives (default on) — extend _parse_method, add _compute_background_stats, branch in _compute_pixel_thresholds_from_volume; new pass_<method>_shape column.
  - id: apply-and-verify
    content: Re-run filter_by_intensity on Sample7_Position7 union variant via LSF; emit filtered TIFF; verify cell 25 removed and strong cells retained.
---

# Per-Channel Intensity QC Filter — Background-Statistics Redesign

## Goal and validation criterion

Replace/augment the current Otsu-based per-channel intensity QC so that cells with near-zero signal in any of the three channels (642, 488, 560) are removed.

Validation target on `output/4_24_25_CGN_6_10_2/Sample7_Position7_decon_dsr/`:

- `union_488_560` label **25** (`mean_488 = 7.4e-6`, near noise floor) **must be filtered OUT** under the chosen method.
- `filtered_642` label **25** (`mean_488 = 1201`, strong on all channels) **must remain IN**.
- Top-ranked retained cells across both variants should be visually plausible (spot-check 5).

## Why current QC fails (union_488_560)

`log_otsu` collapses to **0.0** for both 642 and 488 on the union variant (`threshold_488=0`, `px_threshold_488=0`). Otsu requires bimodality; on this volume the 488 channel is dominated by near-zero values, so the optimizer puts the cut below the noise floor and every cell trivially passes. Cell-aggregated values then have no meaningful gate.

The manager's "background mean + X * std" idea does **not** rely on bimodality — it anchors directly to a noise estimate — so it is the right kind of fix. We will keep it on the evaluation slate but compare it against four close relatives before committing.

## NEW preprocessing rule: clip negatives to zero

Per the manager's note, **negative voxel values are deconvolution artifacts and will be set to 0** before any threshold or per-cell statistic is computed. This applies uniformly to every method evaluated below so the comparison is fair.

Concretely, at the top of `_compute_background_stats`, `_compute_pixel_thresholds_from_volume`, and the per-cell mean/`pct95` recomputation in `run_from_volume`:

```python
combined = np.clip(combined, a_min=0.0, a_max=None)
```

What this changes:

- **Background distribution becomes one-sided (zero-floored).** `bg_mean` shifts up slightly, `bg_std` shrinks. `MAD` becomes a more attractive scale estimator than `std` (`bg_mad` may beat `bg_sigma`).
- **`log_otsu` is unaffected** — it already filters `values > 0`.
- **`otsu` / `otsu2`** behaviour shifts — the lower tail is flattened to 0, making the bimodal cut easier to find when a real foreground exists.
- **Existing `mean_<ch>` in `qc_features.csv`** (written by `features/extract_features.py`) is left raw — the new filter recomputes its own clipped statistics from the variant's `combined.tif` and writes them in the new pass columns; we don't rewrite the upstream feature CSV. This decision is documented in the explore script header.
- **Cell 25 (union)** still removed: `median_488 = 5e-9` and `mean_488 = 7e-6` survive clipping unchanged (already non-negative), but `bg_mean + X * bg_std` lifts to a real-signal scale, dropping it cleanly.

A `--clip-negatives` CLI flag will be added in Phase 2 (default **on**); the explore script in Phase 1 will report a "clip vs no-clip" sensitivity row so we can confirm the answer is robust.

## Phase 0 — Git: snapshot current state, branch off

Current branch: `full-z-range-dev`. **No remote configured** (`git remote -v` empty), so push is deferred unless you add one.

1. Stage and commit on `full-z-range-dev`:
   - All ~30 modified files.
   - Untracked to **include**: `AGENTS.md`, `pipeline-context.md`, `docs/`, `postprocess/__init__.py`, `postprocess/union_488_560_mask.py`, `postprocess/tests/`, otsu2 code+launchers+docs (`qc/*otsu2*`, `OTSU2_*.md`, `DELIVERY_SUMMARY.md`, `OTSU2_FILES_SUMMARY.txt`), `qc/apply_qc_pass_to_label_mask.py`, `qc/filter_full_z_by_pass_pixel_intensity.py`, `qc/merge_qc_features_filtered.py`, `features/compute_dinov2_volume_norm_csv.py`, `features/dinov2/`, `features/extract_dinov2_embeddings.py`, `features/test_crop_cells_full_z.py`, `visualization/__init__.py`, `visualization/dinov2_*.py`, `scripts/`, `.cursor/plans/`, `.cursor/rules/`, `.cursor/clip488_p9999_experiment.md`. (`segmentation_multiscale_cellpose_3D/` is in `archive/legacy_modules/` per `pipeline-context.md`; if it's tracked-untracked here, exclude it.)
   - Untracked to **skip**: `logs/`, `cufile.log` (transient/large).
2. Push only if you add a remote afterwards (e.g. `git remote add origin <URL>`); otherwise commit-only.
3. `git checkout -b bg-sigma-filter-dev` off `full-z-range-dev`. All Phase 1+ work happens on this branch.

## Phase 1 — Investigation script (read-only, evaluates 5 methods + 2 baselines)

### Methods to evaluate (per channel, all on clipped voxels)

- **M1 — bg_sigma** *(manager's idea)*: `T = bg_mean + X * bg_std`, `X in {3, 5, 7, 10}`. `pass_ch = (mean_cell >= T)`.
- **M2 — bg_mad**: `T = bg_median + X * 1.4826 * MAD`, `X in {3, 5, 7, 10}`. Robust to one-sided distributions and to bright debris.
- **M3 — otsu_or_bg (hybrid)**: `T = max(log_otsu, bg_mean + X * bg_std)`. Backwards-compatible; only kicks in when Otsu collapses.
- **M4 — bg_frac (per-cell bright-pixel fraction)**: voxel threshold `T_v = bg_mean + X * bg_std`; `pass_ch = (frac_voxels_above_Tv >= min_frac)`, `min_frac in {0.10, 0.25}`. Robust to the "dilution by big mask" failure mode that traps cell 25 on the union variant.
- **M5 — bg_pct95**: `pass_ch = (pct95_cell >= bg_mean + X * bg_std)`. Cheap; uses `pct95_<ch>` already in `qc_features.csv`.

### Baselines

- `log_otsu` (current default), `otsu2` (already merged in [qc/filter_by_intensity.py](qc/filter_by_intensity.py)).

### Background voxel definition

For each variant + channel:

- Take complement of the cell mask after **dilating it by 2 voxels** (avoids cell-edge bleed).
- After clipping the volume to ≥ 0, optionally drop the **top 1%** of background values (autofluorescent debris). Report stats with and without this clip.

Compute and report `(bg_mean, bg_std, bg_median, MAD)` for each combination (clipped/raw × dilated/raw → 4 background-stat sets per channel).

### Deliverable

New file: [qc/bg_threshold_explore.py](qc/bg_threshold_explore.py)

- CLI: `--output-dir <sample>`, `--variant {filtered_642,union_488_560}`, `--no-clip` (sensitivity), `--report-csv <path>`.
- Loads `<variant>.tif` and `<variant>_combined.tif`; clips negatives to 0 unless `--no-clip`.
- For each method × X (× min_frac for M4), records `threshold` per channel, `n_pass`, and **`cell25_pass` flag**.
- Output report: `<sample>/bg_threshold_report.csv` with columns `method, X, min_frac, channel, threshold, n_pass, cell25_pass, clip_negatives, bg_dilate, bg_top_clip_pct`.
- Console table: per-method-X summary plus an explicit "cell 25 dropped" yes/no column.

LSF launcher: [qc/bsub_bg_threshold_explore.sh](qc/bsub_bg_threshold_explore.sh)

- Standard queue, `-M 16000 -R "rusage[mem=16000]"`, ~1 h walltime, no email flags.
- Absolute paths via `$PROJECT_ROOT`; runs the explorer for both `union_488_560` and `filtered_642`.
- Logs to `logs/bg_threshold_explore_<jobid>.log`.

I will monitor with `bjobs` / log tail and summarize.

**Decision point:** we pick the winning method together. Acceptance gates:

1. `cell25_pass = False` for `union_488_560`.
2. `cell25_pass = True` for `filtered_642` (positive-control cell).
3. Total retained-cell count is in a sensible range (not zero, not unchanged).

## Phase 2 — Productionize the chosen method

Edit [qc/filter_by_intensity.py](qc/filter_by_intensity.py) to add the chosen method(s) without disturbing existing `log_otsu` / `otsu` / `otsu2` paths.

- Extend `_parse_method` to recognize:
  - `bg_sigma:X` → `("bg_sigma", X)`
  - `bg_mad:X` → `("bg_mad", X)`
  - `otsu_or_bg:X` → `("otsu_or_bg", X)`
  - (only if M4 wins) `bg_frac:X:F` → `("bg_frac", (X, F))`
- Add helper `_compute_background_stats(combined, mask, *, clip_negatives=True, dilate_iters=2, top_clip_pct=1.0)` returning per-channel `{bg_mean, bg_std, bg_median, mad}`.
- In `_compute_pixel_thresholds_from_volume(...)` and the per-cell threshold path of `run_from_volume`:
  - clip combined to ≥ 0 when `clip_negatives` is on,
  - branch on `method` to compute `T` from bg stats,
  - for `otsu_or_bg`, take `max(log_otsu_T, bg_floor)` per channel.
- Pass column naming follows the existing otsu2 pattern (`method_suffix`):
  - new `pass_<method>_shape` column (e.g. `pass_bg_sigma_shape`).
  - update `fieldnames` and the summary print accordingly.
- New CLI flags:
  - `--clip-negatives / --no-clip-negatives` (default **on**).
  - `--bg-dilate-iters` (default 2).
  - `--bg-top-clip-pct` (default 1.0).
- The legacy `mean_<ch>` columns continue to be computed on the raw volume so existing CSVs stay comparable; the clipping only affects threshold/pass logic. The explore-script header documents this asymmetry.

No changes to [qc/apply_qc_pass_to_label_mask.py](qc/apply_qc_pass_to_label_mask.py), [postprocess/__init__.py](postprocess/__init__.py), segmentation, or feature-extraction code.

## Phase 3 — Apply on the test sample, regenerate filtered TIFF

- LSF-run `python qc/filter_by_intensity.py --from-volume --variant union_488_560 --method <winner> --clip-negatives --output-dir <sample>` (~16 GB).
- Generate `union_488_560_<method>_filtered.tif` by either:
  - generalising [qc/create_otsu2_shape_filtered_mask.py](qc/create_otsu2_shape_filtered_mask.py) to take `--pass-column` + `--output-name`, or
  - writing a thin `qc/create_pass_filtered_mask.py` wrapper. (Decide based on whether we want to retire the otsu2-specific helper.)
- Verify:
  - cell 25 zeroed-out in the output mask;
  - cell counts comparable to the existing `pass_otsu_shape` baseline (with the expected reduction);
  - 5 spot-checked high-intensity cells still present.
- Brief written summary in `docs/` (`docs/bg_sigma_filter_results.md`) following the otsu2 doc style.

## Open knobs (defaults I'll use unless you say otherwise)

- `X` sweep: {3, 5, 7, 10} for sigma- and MAD-based methods.
- Background dilation: 2 voxels.
- Background top-clip: 1%.
- `min_frac` for M4: {0.10, 0.25}.
- Negative clipping: ON for thresholds and pass logic; legacy `mean_<ch>` columns stay raw.

## File map (proposed)

- New (Phase 1): [qc/bg_threshold_explore.py](qc/bg_threshold_explore.py), [qc/bsub_bg_threshold_explore.sh](qc/bsub_bg_threshold_explore.sh).
- Modified (Phase 2): [qc/filter_by_intensity.py](qc/filter_by_intensity.py).
- New (Phase 3): `qc/bsub_bg_filter_test.sh`, possibly `qc/create_pass_filtered_mask.py`, `docs/bg_sigma_filter_results.md`.
- Untouched: [qc/apply_qc_pass_to_label_mask.py](qc/apply_qc_pass_to_label_mask.py), [postprocess/__init__.py](postprocess/__init__.py), all segmentation / feature / visualization code, [qc/create_otsu2_shape_filtered_mask.py](qc/create_otsu2_shape_filtered_mask.py) (kept for reference unless we explicitly replace it).
