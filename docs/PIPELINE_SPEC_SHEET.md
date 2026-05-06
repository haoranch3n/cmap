# CMAP analysis pipeline — specification sheet

**Audience:** managers and collaborators who need a clear, non-code overview of what the pipeline does, which algorithms are used, and how results reach Napari.

**Scope:** From per-cell image crops (“cell boxing”) through QC, optional DINOv2 embeddings, UMAP sweep, publication of a single “best” 2D layout, and Napari review. Segmentation is summarized only at a high level.

**Where outputs live:** By default under the repository’s `output/` directory. Your site may set `**PIPELINE_OUTPUT_DIR`** (or `**CMAP_OUTPUT_DIR**`) so all stages write to the same root; Napari must be pointed at that same root.

---

## 1. High-level flow


| Stage                         | Module / script (typical)                                            | Purpose                                                               |
| ----------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Segmentation                  | `segmentation/run_pipeline.py`                                       | Raw volume → 3D cell label masks                                      |
| Postprocess                   | `postprocess/filter_642_mask.py`, `postprocess/combine_with_mask.py` | Clean / combine masks with intensity data for downstream crops        |
| Cell crops (“boxing”)         | `features/crop_cells.py`                                             | Cut out each cell as a small multi-channel TIFF                       |
| Classical QC features         | `features/extract_features.py`                                       | Per-cell scalar measurements (e.g. channel means) → `qc_features.csv` |
| Intensity QC                  | `qc/filter_by_intensity.py`                                          | Add pass/fail columns → `qc_features_filtered.csv`                    |
| Optional: filtered crops      | Separate QC/crop steps (e.g. filtered boxing dirs)                   | Restrict analysis to cells that pass QC                               |
| Optional: DINOv2              | `features/extract_dinov2_embeddings.py`                              | Deep neural embedding per cell → `dinov2_embeddings.npy` + `.csv`     |
| Optional: UMAP on pooled DINO | `visualization/dinov2_umap_grid_search.py`                           | Many 2D UMAPs + scores; merge + pick winner                           |
| Publish for Napari            | Same script, `--publish-best`                                        | One master table with the chosen 2D coordinates                       |
| Interactive QC                | `napari-plugin/`                                                     | Load cell TIFFs + master table; explore embedding                     |


End-to-end orchestration (when used): `pipelines/run_analysis_pipeline.py` (with optional `--run-dinov2`, `--also-full-z`, etc.).

---

## 2. Cell boxing (cropping each cell)

**Input:** Combined / mask volumes after postprocess (script discovers paths under the sample output directory).

**How a cell’s spatial extent is found:** For each integer **label** in the 3D mask, the pipeline uses **scikit-image `regionprops`**, which provides an axis-aligned **bounding box** around that label.

**How the crop is built:** The bounding box is expanded by fixed **margins** (defaults: 20 voxels in XY, 5 in Z, configurable in `crop_cells.py`), clipped to the volume, and exported as a **Z × channels × Y × X** TIFF per cell (`cell_*.tif`). Channels typically include the fluorescent channels plus a primary mask channel and a boundary channel for QC geometry.

**Takeaway for a manager:** The “box” is the mask-derived bounding box (not a centroid-based box). Centroids appear later, only in the **DINOv2** path (Section 5).

---

## 3. Classical QC features and intensity filtering

**Features (`extract_features.py`):** Reads each cell TIFF and writes `**cell_qc/qc_features.csv`** with per-cell numeric descriptors (e.g. mean intensity per channel and related stats used downstream).

**Filtering (`filter_by_intensity.py`):** Produces `**cell_qc/qc_features_filtered.csv`** with extra columns:

- **Per-channel mean thresholds** — Default method for cell-level means is `**log_otsu`** (log transform of positive intensities, then **Otsu’s method** from scikit-image) unless overridden (`--method`: `otsu`, `percentile:N`, `fixed:V`).
- **Per-channel “pixel” thresholds** — Computed from **all masked pixels** pooled across cell TIFFs in `cell_boxing/`; by default **log + Otsu** on positive pixels.
- **Pass flags** — `pass_*`, `pass_intensity`, `pass_pixel_intensity`, etc., encode which cells pass which rules.

**Takeaway:** “Filtering” here is **rule-based QC on measured intensities**, not deep learning. It decides which cells are trustworthy for pooled analysis and optional filtered-only folders.

---

## 4. Optional: filtered cell boxing

If your workflow copies or regenerates crops only for cells that pass QC (e.g. under `**cell_boxing_filtered/`** with sibling `**cell_qc_filtered/**`), downstream DINOv2 and Napari can target **that** subset. The exact copy/filter step depends on which site scripts you run; the important idea is **the same `cell_id` / `sample` / `dataset` keys** must stay aligned so tables and TIFFs join correctly.

---

## 5. DINOv2 embeddings (optional but required for DINO UMAP)

**Model:** Frozen **Meta DINOv2 ViT-B/14** via PyTorch Hub — by default a **768-dimensional** vector per cell (CLS token) per encoded view.

**Image normalization:** Fluorescence-friendly **percentile clipping** per channel (defaults 1st–99th percentile, **slice** or **volume** scope). ImageNet normalization is **off** by default for microscopy.

**How the “centroid” is determined (orthogonal mode only):** When extraction mode is `**orthogonal_concat`**, the code does **not** use the mask bbox center. It computes a **robust 3D centroid** inside the cell mask:

1. Compute the **Euclidean distance transform** (distance from each foreground voxel to the nearest background voxel).
2. Keep voxels whose distance is in the **upper quantile** of distances over the foreground (default first try: **75th percentile** — an “interior core” that ignores thin bridges to neighbors).
3. Take the **center of mass** of that core, optionally **weighted by total intensity** across the three fluorescent channels (default: **on**).
4. If the core is too small, the algorithm **relaxes** the quantile (e.g. 0.55, 0.35, 0.15) and finally falls back to the full-mask center of mass.

**Why:** To avoid a centroid sitting on a **merge bridge** or edge artifact between touching cells.

**How slices enter the network:**  

- `**z_mean` (default):** All Z planes are encoded; embeddings are **mean-pooled** → one **768-D** vector per cell.  
- `**orthogonal_concat`:** Three **2D planes** through the robust centroid (XY at fixed Z, XZ at fixed Y, YZ at fixed X) are encoded; the three 768-D vectors are **concatenated** → **2304-D** per cell (if that mode is enabled).

**Outputs (per sample):** e.g. `**cell_qc/…/dinov2_embeddings.npy`** and `**dinov2_embeddings.csv**` (metadata + paths). Filtered-tree variants use parallel directory names when configured.

---

## 6. Pooling embeddings across samples / datasets

**Script helpers:** `visualization/dinov2_pool.py`.

**What “pooling” means:** **Vertical stacking** of per-sample embedding matrices into one matrix **X** of shape **(N, D)** where **N** is the total number of cells and **D** is the embedding dimension. Metadata rows (dataset, sample, `cell_id`, etc.) are concatenated in the **same order** so each row of **X** matches one metadata row.

**Takeaway:** UMAP is run on **one joint cloud of all pooled cells**, so the 2D layout is **comparable across samples** in that run.

---

## 7. UMAP hyperparameter sweep

**Script:** `visualization/dinov2_umap_grid_search.py`.

**Algorithm:** `**umap-learn`** `UMAP`, **2 components**, fixed `**random_state=42`**.

**Default grid (12 combinations):**


| Hyperparameter | Values tried      |
| -------------- | ----------------- |
| `n_neighbors`  | 5, 15, 30         |
| `min_dist`     | 0.0, 0.1          |
| `metric`       | cosine, euclidean |


**Note:** If a requested `n_neighbors` is larger than **N − 1**, it is **capped** automatically.

**Per combo outputs:**  

- A coordinate CSV (`dinov2_umap_coords_*.csv`) with `**umap_1` / `umap_2`** plus metadata.  
- A JSON score file (`dinov2_umap_score_*.json`).

---

## 8. How the “final” UMAP is chosen

**Intrinsic quality metric:** For each hyperparameter combination, the pipeline scores the 2D embedding against the **high-dimensional** pooled matrix **X** using **sklearn’s `trustworthiness`** — a standard measure of how well local neighborhoods in high-D are preserved in 2D.

**Two neighborhood sizes are recorded:** `**trustworthiness_k5`** and `**trustworthiness_k15**` (k is capped by **N − 1**).

**Default winner rule:** `**--publish-best`** selects the combination with the **maximum `trustworthiness_k15`**. Rationale: k=15 is slightly more **global** than k=5, so it favors layouts that preserve structure beyond immediate nearest neighbors, while still reflecting local fidelity. (You can override the column with `**--score-column`** if your group agrees on a different criterion.)

**Steps operators run:**

1. `**--merge-scores`** (or implicit merge) → `cell_qc_all/umap_dinov2_sweep/dinov2_umap_sweep_scores.csv`
2. `**--publish-best**` → writes `**cell_qc_all/dinov2_embedding_all.csv**` (and `**dinov2_embeddings_all.npy**`) with columns `**umap_dinov2_1` / `umap_dinov2_2**` copied from the winning combo’s `umap_1` / `umap_2`, plus bookkeeping columns (e.g. which slug won and the score value).

---

## 9. Input to Napari

**Plugin:** `napari-plugin/` (install editable, open the **same** output root the pipeline used).

**Master table discovery:** Under `**cell_qc_all/`**, if `**dinov2_embedding_all.csv**` exists, the plugin **prefers it** as the master feature/embedding CSV (so the published DINO UMAP is not skipped in favor of older classical master tables).

**Joining rows to TIFFs:** Each loaded cell image is matched to a CSV row by `**sample` + `cell_id`**, and by `**dataset**` as well when that column exists (multi-batch outputs).

**What you see after “Load Embedding”:** The first available pre-computed pair is used in this order: **DINOv2 UMAP (sweep best)** → classical UMAP variants. So after publish, the **published** `umap_dinov2_*` columns load by default.

**Clustering in Napari:** **Leiden** (or fallbacks) runs on the **displayed 2D coordinates** so cluster colors match what you see. That choice does **not** change which UMAP was selected; it only groups points in the current view.

**Cell image display vs analysis paths:** If the session lists TIFFs under `cell_boxing_filtered/` but your output tree also has **`cell_boxing_full_z_filtered/`** with the **same filenames** (`cell_XXXX.tif`), Napari loads the **full-Z** file **only for the image viewer**. Feature joins, DINOv2 inputs, UMAP, and annotation keys remain tied to the **filtered** paths you opened.

---

## 10. One-page summary (elevator pitch)

1. **Masks → boxes:** Each cell is cropped with a **mask bounding box + margin**.
2. **QC:** Classical measurements + **Otsu / log-Otsu** thresholds mark cells that pass intensity checks.
3. **DINOv2 (optional):** A frozen **ViT-B/14** encoder embeds each cell; in orthogonal mode a **distance-transform “interior” centroid** picks three planes.
4. **Pooled UMAP:** All chosen cells are stacked; **UMAP** is run for a **small grid** of hyperparameters.
5. **Winner:** Default **max trustworthiness at k=15** picks one combo; `**--publish-best`** writes `**dinov2_embedding_all.csv**`.
6. **Napari:** Opens that master file on the same disk root so reviewers explore the **official** 2D layout and QC cells interactively.

---

*This document reflects the CMAP repository layout as of its last update; CLI defaults may be overridden at your site. For path and orchestration details, see `pipeline-context.md` and `MODULE_ARCHITECTURE.md`.*