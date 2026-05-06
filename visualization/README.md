# Visualization Module

This module owns embedding/plotting/report generation from QC feature tables.

Current implementation runs natively here with code migrated from:
- `cell_qc/tsne_visualize.py` -> `visualization/tsne_visualize.py`
- `cell_qc/tsne_visualize_all.py` -> `visualization/tsne_visualize_all.py`

## DINOv2 embedding pooling (optional)

`visualization/dinov2_visualize.py` walks the output tree for
`<cell-qc-dir>/dinov2_embeddings.npy` + `dinov2_embeddings.csv` (default
`<cell-qc-dir>=cell_qc`; use `--cell-qc-dir cell_qc_filtered` for filtered
crops). It **concatenates** every cell row from every sample into one matrix
`(N, D)` with `N` the total cell count — **each row is still one cell** for
UMAP (no averaging across cells). Optionally filters to QC-passing cells via
`qc_features_filtered.csv` (`--filter-col`), and writes
`cell_qc_all/dinov2_embeddings_all.npy` plus a UMAP-augmented
`dinov2_embedding_all.csv`.

Shared discovery/loading lives in `visualization/dinov2_pool.py`.

UMAP defaults (`n_neighbors=5`, `min_dist=0.0`, `metric=cosine`) match the
historical napari convention referenced in older docs.

## DINOv2 UMAP hyperparameter sweep (4_18 + 4_24 combined)

`visualization/dinov2_umap_grid_search.py` loads **all** per-cell embeddings from
`output/4_18_25/**/cell_qc_filtered/` and `output/4_24_25_CGN_6_10_2/**/cell_qc_filtered/`,
fits UMAP for one hyperparameter triple (`--n-neighbors`, `--min-dist`,
`--metric`), and writes:

- `output/cell_qc_all/umap_dinov2_sweep/dinov2_umap_coords_<slug>.csv`
- `output/cell_qc_all/umap_dinov2_sweep/dinov2_umap_score_<slug>.json`

LSF batching: `scripts/batch_dinov2_umap_sweep/` (standard queue, one job per
combo). After jobs finish, run `--merge-scores` to build
`dinov2_umap_sweep_scores.csv`.

Then publish the winning 2D layout (default: maximize `trustworthiness_k15`)
into the Napari master files:

```bash
python visualization/dinov2_umap_grid_search.py --publish-best --output-root "$PIPELINE_OUTPUT_DIR"
```

This writes `cell_qc_all/dinov2_embedding_all.csv` and `dinov2_embeddings_all.npy`
with columns `umap_dinov2_1` / `umap_dinov2_2`. Override the score with
`--score-column`.

Intrinsic quality columns in each JSON: `trustworthiness_k5`,
`trustworthiness_k15` (sklearn, comparing high-D `X` to 2D embedding).
