# Processing Pipeline: Steps, Parameters, and Skip Behavior

## Pipeline Steps Overview

The processing pipeline in `ProcessImages()` (see `processing_utils.py`) executes the following steps in order.
The table below shows which steps run depending on the **Skip Segmentation** checkbox
and which **parameter** controls each step.

| #  | Step                              | Skip Seg OFF (full) | Skip Seg ON         | Controlled By              |
|----|-----------------------------------|---------------------|---------------------|----------------------------|
| 1  | Background estimation (percentile filter) | **RUN**       | SKIP (load saved)   | `BKG Percentile`           |
| 2  | Background subtraction (GPU)      | **RUN**             | SKIP (load saved)   | `BKG Percentile`           |
| 3  | Normalization (GPU)               | **RUN**             | SKIP (load saved)   | `BKG Percentile`           |
| 4  | Save normalized + subtracted images | **RUN**           | SKIP                | `BKG Percentile` (in filename) |
| 5  | ilastik pixel classification      | **RUN**             | SKIP (load saved)   | `ilastik Model`            |
| 6  | Save segmented image              | **RUN**             | SKIP                | `BKG Percentile` (in filename) |
| 7  | Temporal color coding             | **RUN**             | **RUN**             | (none)                     |
| 8  | Connected component labeling      | **RUN**             | **RUN**             | (none)                     |
| 9  | Region measurement (per time point) | **RUN**           | **RUN**             | (none)                     |
| 10 | Save initial region CSV           | **RUN**             | **RUN**             | `BKG Percentile` (in filename) |
| 11 | Area filtering                    | **RUN**             | **RUN**             | `Min Area`, `Max Area`     |
| 12 | Hierarchical clustering (Ward)    | **RUN**             | **RUN**             | `Max Cluster Distance`     |
| 13 | IoU-based cluster merging         | **RUN**             | **RUN**             | `IoU Threshold` (default 0.4, not in UI) |
| 14 | Map clusters to image mask + save | **RUN**             | **RUN**             | (none)                     |
| 15 | Region measurement (on final mask)| **RUN**             | **RUN**             | (none)                     |
| 16 | Save full region CSV              | **RUN**             | **RUN**             | `BKG Percentile` (in filename) |
| 17 | SNR filtering + save accepted/rejected masks | **RUN**  | **RUN**             | `SNR Threshold`            |


## What Happens When You Change Each Parameter

### Skip Segmentation (checkbox)

| Setting | Behavior |
|---------|----------|
| **OFF** (unchecked) | Full pipeline: steps 1-17 all execute. Background subtraction, normalization, and ilastik segmentation are computed from scratch and saved. |
| **ON** (checked) | Steps 1-6 are skipped. Previously saved normalized, subtracted, and segmented images are loaded from disk. Steps 7-17 (labeling, clustering, filtering) are **always re-run**. |

**When to use Skip Segmentation ON:** When you have already run full processing once
and want to re-run only the downstream analysis (e.g., with different clustering distance)
without redoing the expensive background subtraction and ilastik steps.

**Requirement:** The output folder must already contain the files from a previous full run
(in `BKG_subtracted_normalized/`, `BKG_subtracted/`, and `Segmented/` subfolders).

---

### Background Percentile (`BKG Percentile`)

- **Default:** 10
- **Used in steps:** 1-4 (background estimation, subtraction, normalization)
- **Also affects:** File naming for ALL output files (filenames contain `percentile10`, etc.)

| Skip Seg | Effect of changing this value |
|----------|-------------------------------|
| OFF | Background is recomputed with the new percentile. All outputs are saved with the new percentile in filenames (new files, old ones are NOT overwritten). |
| ON  | The code looks for previously saved files with the new percentile in the filename. **If those files don't exist (because the previous run used a different percentile), processing will FAIL.** You must run with Skip Segmentation OFF first with the new percentile value. |

---

### Max Cluster Distance (`Max Cluster Distance`)

- **Default:** 6 pixels
- **Used in step:** 12 (hierarchical clustering)
- **Passed to `ProcessImages`:** Yes

| Skip Seg | Effect of changing this value |
|----------|-------------------------------|
| OFF | Full pipeline runs. Clustering uses the new distance. |
| ON  | Steps 1-6 are skipped (loaded from disk). Clustering in step 12 uses the new distance. **This is the ideal use case for Skip Segmentation** -- re-cluster without redoing expensive preprocessing. |

---

### Min Area / Max Area

- **Shown in viewer UI:** Yes (Min Area default: 20, Max Area default: 400)
- **Used in step:** 11 (area filtering)
- **Passed to `ProcessImages`:** Yes

| Skip Seg | Effect of changing this value |
|----------|-------------------------------|
| OFF | Full pipeline runs. Objects outside the area range are excluded before clustering. |
| ON  | Steps 1-6 skipped. Area filtering in step 11 uses the new values. **Good use case for Skip Segmentation** -- change area filter without redoing preprocessing. |

---

### SNR Threshold

- **Shown in viewer UI:** Yes (default: 3.0)
- **Used in step:** 17 (SNR filtering)
- **Passed to `ProcessImages`:** Yes

| Skip Seg | Effect of changing this value |
|----------|-------------------------------|
| OFF | Full pipeline runs. Clusters below the new SNR threshold are rejected. |
| ON  | Steps 1-6 skipped. SNR filtering in step 17 uses the new threshold. **Good use case for Skip Segmentation** -- adjust acceptance criteria without redoing preprocessing. |

---

### SNR Start Frame / SNR End Frame

- **Configured in:** `config.py` (`DEFAULT_SNR_START_FRAME`, `DEFAULT_SNR_END_FRAME`)
- **Not shown in viewer UI** -- edit `config.py` to change.
- **Used in step:** 17 (SNR filtering -- controls which frames of each trace are used for baseline/signal calculation)
- **Passed to `ProcessImages`:** Yes

| Skip Seg | Effect of changing these values |
|----------|--------------------------------|
| OFF | Full pipeline runs. SNR is calculated using only the specified frame range of each cluster's trace. |
| ON  | Steps 1-6 skipped. SNR calculation uses the new frame range. **Good use case for Skip Segmentation** -- recalculate SNR on a different portion of the recording without redoing preprocessing. |

Setting both to `None` uses the entire trace.
These are useful when only a specific portion of the recording should be considered
for determining signal quality (e.g., a known baseline period).

---

### ilastik Model

- **Used in step:** 5 (pixel classification)
- **Model type** is auto-detected from the filename (`spon`/`spononly` = zyx, otherwise = tyx)

| Skip Seg | Effect of changing the model |
|----------|------------------------------|
| OFF | The new model is used for segmentation. New segmented image is saved. |
| ON  | The model is **not used** -- the previously saved segmented image is loaded instead. Changing the model has no effect unless you uncheck Skip Segmentation. |

---

### IoU Threshold (not in viewer UI)

- **Default:** 0.4
- **Configured in:** `iglusnfr_config.json` → `processing.iou_threshold`
- **Used in step:** 13 (IoU-based cluster merging)
- Can also be set by calling `ProcessImages()` directly from a script with `iou_threshold=<value>`.

---

### Parallel Workers (`n_jobs`)

- **Default:** 15
- **Configured in:** `iglusnfr_config.json` → `processing.n_jobs`
- **Used in steps:** 9, 15 (region measurement via `joblib.Parallel`)
- Reduce if you run out of memory. Increase if you have many CPU cores.

### Parallel Backend (`parallel_backend`)

- **Default:** `"threading"`
- **Configured in:** `iglusnfr_config.json` → `processing.parallel_backend`
- Options: `"threading"` (shared memory, lower overhead) or `"loky"` (process-based, better isolation).


## Summary: Which Steps Re-run When Re-processing

```
Skip Segmentation OFF (full processing):
  [1] BKG estimation    -> RUN (uses BKG Percentile)
  [2] BKG subtraction   -> RUN
  [3] Normalization      -> RUN
  [4] Save norm+sub      -> RUN
  [5] ilastik classify   -> RUN (uses Model)
  [6] Save segmented     -> RUN
  [7] Color coding       -> RUN
  [8] Component labeling -> RUN
  [9] Region measurement -> RUN
  [10] Save initial CSV  -> RUN
  [11] Area filtering    -> RUN (uses Min Area, Max Area)
  [12] Clustering        -> RUN (uses Max Cluster Distance)
  [13] IoU merging       -> RUN (uses IoU Threshold, default 0.4)
  [14] Save cluster mask -> RUN
  [15] Region measure    -> RUN
  [16] Save full CSV     -> RUN
  [17] SNR filtering     -> RUN (uses SNR Threshold)

Skip Segmentation ON (re-process downstream only):
  [1] BKG estimation    -> SKIP (load from BKG_subtracted_normalized/)
  [2] BKG subtraction   -> SKIP (load from BKG_subtracted/)
  [3] Normalization      -> SKIP (loaded above)
  [4] Save norm+sub      -> SKIP
  [5] ilastik classify   -> SKIP (load from Segmented/)
  [6] Save segmented     -> SKIP
  [7] Color coding       -> RUN  *
  [8] Component labeling -> RUN  *
  [9] Region measurement -> RUN  *
  [10] Save initial CSV  -> RUN  * (overwrites previous)
  [11] Area filtering    -> RUN  * (uses Min Area, Max Area)
  [12] Clustering        -> RUN  * (uses Max Cluster Distance)
  [13] IoU merging       -> RUN  * (uses IoU Threshold)
  [14] Save cluster mask -> RUN  * (overwrites previous)
  [15] Region measure    -> RUN  *
  [16] Save full CSV     -> RUN  * (overwrites previous)
  [17] SNR filtering     -> RUN  * (uses SNR Threshold, overwrites previous)

  * = always re-runs regardless of parameter changes
```


## Notes

1. **Changing `BKG Percentile` with Skip Segmentation ON will fail** because the
   code looks for files with the new percentile value in the filename, which
   won't exist if the previous run used a different percentile.

2. **Steps 7-17 always re-run** when Skip Segmentation is ON. There is no way
   to skip just the clustering or just the SNR filtering independently.

3. **Output files are overwritten** when re-processing with Skip Segmentation ON.
   The downstream outputs (CSVs, masks, plots) from the previous run are replaced.

4. **IoU Threshold** (default 0.4) is accepted by `ProcessImages()` but not
   exposed in the viewer UI. To change it, call `ProcessImages()` directly.
