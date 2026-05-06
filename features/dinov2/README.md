# DINOv2 cell-embedding subpackage

Frozen DINOv2 ViT-B/14 encoder run as a 2.5D feature extractor over the
existing CMAP cell crops. Slots in alongside `features/extract_features.py`
without changing any earlier stage.

## Inputs

Reads `output/<sample>/cell_boxing/cell_<NNNN>.tif` exactly as written by
`features/crop_cells.py`:

- shape `(Z, 5, Y, X)` (older 4-channel and raw 3-channel files are
  tolerated)
- channels `[642, 488, 560, Primary_Cell_Mask, Mask_Boundary]`

Only the first three intensity channels are sent to DINOv2. The mask channel
is consumed only when `--apply-mask` is set.

## Outputs (per sample)

Written under `output/<sample>/cell_qc/`:

- `dinov2_embeddings.npy` — float32 array of shape `(N_cells, 768)`
- `dinov2_embeddings.csv` — keyed by `cell_id`; one row per embedding plus
  metadata (`embedding_dim`, `n_valid_slices`, `original_z/y/x`, `model`,
  `target_size`, `apply_mask`, `use_imagenet_stats`, `norm_p_low`,
  `norm_p_high`, `empty_slice_nonzero_frac`)
- `dinov2_extract_summary.txt` — short run summary
- `dinov2_embeddings_per_slice.npz` — only with `--save-slice-embeddings`,
  a per-cell map `cell_<NNNN>` → `(N_valid, 768)` for debugging /
  attention-pooling experiments later

The pooled cross-sample artefacts live under
`output/cell_qc_all/dinov2_embeddings_all.npy` and
`dinov2_embedding_all.csv`; see `visualization/dinov2_visualize.py`.

## Defaults and how to swap them later

- **Loading path**: `torch.hub.load("facebookresearch/dinov2",
  "dinov2_vitb14")`. Stable, no `timm` dependency.
- **Embedding output**: the default `forward()` returns the **CLS token**.
  ViT-B/14 → 768-D vector per slice. To switch to patch tokens (for
  attention pooling), add a method on `DinoV2Encoder` and call
  `model.forward_features(...)`.
- **Normalization**: per-slice, per-channel 1st–99th percentile clip →
  rescale to `[0, 1]`. Microscopy intensities span several orders of
  magnitude and ImageNet stats do not transfer cleanly, so we keep that off
  by default and expose `--imagenet-stats` for users who want to layer it
  on top.
- **Resize**: 224×224 (multiple of 14). Configurable via `--target-size`.
- **Aggregation**: mean over valid z-slices. Max pooling lives next to it
  in `aggregate.py`; attention pooling is the natural next addition there.
- **Channel mapping**: `[642, 488, 560]` → R, G, B in input order. We do
  not reorder for "DAPI/cyto1/cyto2" semantics — 642 is the segmentation
  channel in this pipeline, not DAPI.

## Example commands

Run on a single sample and override the device / batch size:

```bash
python features/extract_dinov2_embeddings.py \
    --data-rel CGNSample1_Position0_decon_dsr \
    --batch-size 32 --device cuda
```

Mask out neighbour cells before encoding, and dump per-slice vectors:

```bash
python features/extract_dinov2_embeddings.py \
    --data-rel CGNSample1_Position0_decon_dsr \
    --apply-mask --save-slice-embeddings
```

Run on an explicit pair of dirs (no `data/` mirror):

```bash
python features/extract_dinov2_embeddings.py \
    --data-dir /path/to/data --output-dir /path/to/output
```

Pool everything and run UMAP, restricted to QC-passing cells:

```bash
python visualization/dinov2_visualize.py \
    --filter-col pass_intensity --suffix pxfiltered
```

End-to-end via the orchestrator (after segmentation/postprocess/features/qc):

```bash
python pipelines/run_analysis_pipeline.py \
    --skip-segmentation --run-dinov2 --run-dinov2-vis \
    --dinov2-extra --device cuda --batch-size 32
```

## Running the tests

```bash
pytest features/dinov2/tests/test_dinov2_pipeline.py -q
```

The tests use a tiny CPU-only stub encoder; no network or GPU required.

## Extension points

- `aggregate.py` — add `attention_pool` next to `mean_pool`/`max_pool`.
- `encoder.py` — add a `encode_patch_tokens` method using
  `model.forward_features(...)["x_norm_patchtokens"]` once attention pooling
  needs them.
- A linear-probe trainer can be added under `features/dinov2/` consuming
  the same `dinov2_embeddings.npy` matrices.
- A MIP baseline can sit beside this module: take a max projection across
  z, run DINOv2 once per cell, and dump to `dinov2_mip_embeddings.npy` for
  apples-to-apples comparison.
