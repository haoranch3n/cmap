# segmentation_registered_crop

Segmentation pipeline for registered cell crops produced by Jorge's registration pipeline.

For each registered crop and for each of the three fluorescence channels (642, 488, 560),
the pipeline runs multiscale Cellpose 2D → 3D assembly and selects the single **main cell**
(the one whose centroid is closest to the crop centre). The final output is a
5-channel TIFF per channel per crop, with the original intensities in channels 0–2 and
the segmentation mask + boundary in channels 3–4.

---

## Input

Jorge's registered cell crops:
```
/research/dept/dnb/core_operations/ImageAnalysisScratch/Gutierrez/
  CMAP_general/No_decon_tests/outputs/cell_crops/batch_processing/
    <batch>/
      <Sample>_<Position>/
        <cell_id>/
          <cell_id>_registered.tif    ← shape (Z, C=5, Y, X), float32
          traceability.yaml
```

The registered TIFF has 5 channels:
| Index | Channel | Segmented? |
|---|---|---|
| 0 | 642 nm (structural reference) | ✅ |
| 1 | 488 nm | ✅ |
| 2 | 560 nm | ✅ |
| 3, 4 | derived / not used | ❌ |

---

## Output

```
output_registered_crop_seg/<batch>/<Sample_Position>/<cell_id>/
  <cell_id>_ch642_segmented.tif     ← (Z, 5, Y, X) float32
  <cell_id>_ch488_segmented.tif
  <cell_id>_ch560_segmented.tif
  <cell_id>_ch642_selection.json    ← main-cell metadata
  <cell_id>_ch488_selection.json
  <cell_id>_ch560_selection.json
  ch642/                            ← per-channel working dirs
    tif_planes/
    segmentation_2D_diameters/
    segmentation_2D_planes/
    segmentation_2D_stack/
    segmentation_3D_masks/
  ch488/
  ch560/
```

### 5-channel output format
| Ch | Content |
|---|---|
| 0 | 642 nm intensity (original) |
| 1 | 488 nm intensity (original) |
| 2 | 560 nm intensity (original) |
| 3 | Segmentation mask: 0=background, 1=main cell |
| 4 | Segmentation boundary map (thick, from `skimage.segmentation.find_boundaries`) |

---

## Pipeline steps

For each crop × channel:

1. **Channel extraction** (`preprocessing/extract_channel.py`)
   - Reads `traceability.yaml` → looks up full-FOV normalization bounds from
     `output_no_deconv/<experiment>/<Sample>_Position<N>/<ch>nm_crop/tif_planes/segmentation_norm_bounds.csv`
   - Applies affine normalization: `(x - lo) / (hi - lo)` using the full-image percentiles
   - Writes float32 per-Z planes + `segmentation_norm_bounds.csv`

2. **Multiscale Cellpose 2D** (`multiscale_cellpose/segmentation_cellpose_2d.py`)
   - Model: `cpsam` (Cellpose-SAM, v4.0 — latest and most capable)
   - Diameters: 50, 60, 70, 80, 90, 100 px
   - `normalize=False` (pre-normalized in step 1)
   - Merges scales largest→smallest, filters small objects, splits merged cells

3. **Stack 2D → 3D** (`cellcomposor/stack_2d_planes.py`)
   - Stacks per-Z final mask TIFFs into a single `(Z, Y, X)` volume

4. **3D assembly + quality filters** (`cellcomposor/create_3d_cells.py`)
   - Cell matching across Z (Jaccard index ≥ 0.1)
   - Gap bridging, fragment absorption, z-span filter (≥5 slices), volume filter (≥100 vx),
     area-consistency trim, 3D connectivity split

5. **Main cell selection** (`cell_selector/select_main_cell.py`)
   - Computes 3D centroid for each label
   - Selects the label whose centroid is closest to `(Z/2, Y/2, X/2)`
   - Builds binary mask and boundary map

6. **Assemble 5-channel output** (inside `cell_selector/select_main_cell.py`)
   - Stacks original 642/488/560 intensities + mask + boundary into `(Z, 5, Y, X)`

---

## How to run

### On a GPU node (recommended)

```bash
# Submit all batches:
bsub < segmentation_registered_crop/scripts/bsub_run_batch.sh

# Submit a specific batch:
BATCH=4_24_25_CGN_6_10_2 bsub < segmentation_registered_crop/scripts/bsub_run_batch.sh
```

### Manual (on a node with Cellpose + GPU):

```bash
cd /path/to/cmap
conda activate cellpose

# All crops, all channels:
python segmentation_registered_crop/run_pipeline.py

# Specific batch:
python segmentation_registered_crop/run_pipeline.py --batch 4_24_25_CGN_6_10_2

# Specific crop and channel (for testing):
python segmentation_registered_crop/run_pipeline.py \
    --batch 4_24_25_CGN_6_10_2 \
    --crop cell_0002 \
    --channel ch642 \
    --force

# Dry run (just list crops):
python segmentation_registered_crop/run_pipeline.py --dry-run
```

### CLI options

| Flag | Description |
|---|---|
| `--crops-root` | Override registered crops root (default: `$CROPS_ROOT` or Jorge's path) |
| `--output-root` | Override output root (default: `$SEG_CROP_OUTPUT_ROOT` or `output_registered_crop_seg/`) |
| `--batch` | Process only crops in this batch folder |
| `--crop` | Process only this `cell_id` |
| `--channel` | Process only `ch642`, `ch488`, or `ch560` |
| `--force` | Re-run even if outputs already exist |
| `--no-gpu` | Force CPU mode |
| `--dry-run` | List crops without running |

---

## Configuration

Edit `pipeline_config.py` to change:
- `CROPS_ROOT` / `SEG_CROP_OUTPUT_ROOT` / `NORM_BOUNDS_ROOT` (also via env vars)
- `SEGMENTATION_CHANNELS` (channel name → index mapping)
- `CELLPOSE_PRETRAINED_MODEL`, `CELLPOSE_DIAMETERS`, flow/cellprob thresholds
- `MIN_CELL_Z_SPAN`, `MIN_CELL_VOLUME_3D`, `MAX_AREA_CHANGE_RATIO`

---

## Dependencies

Same as `segmentation/`:
```
cellpose>=4.0
numpy
tifffile
scikit-image
scipy
tqdm
pyyaml
```

PyYAML is needed for reading `traceability.yaml`:
```bash
pip install pyyaml
```

---

## Relationship to other modules

| Module | Role |
|---|---|
| `segmentation/` | Active FOV-level segmentation pipeline — source of adapted code |
| `segmentation_multiscale_cellpose_3D/` | Legacy pipeline (reference only) |
| `segmentation_registered_crop/` | This module — per-crop segmentation on 3 channels |
| `postprocess/` | Downstream quantification and filtering (out of scope here) |
