# 99.99% pre-clip for 488 Cellpose + LSF rerun (blocked in Plan mode)

Switch this chat to **Agent** mode and ask to apply this file, or copy the patches below.

## 1. `segmentation_multiscale_cellpose_3D/pipeline_config.py`

After `CELLPOSE_CELLPROB_THRESHOLD = 0.0` add:

```python
def _optional_float_env(key: str):
    """Return float from env, or None if unset/empty."""
    v = os.environ.get(key)
    if v is None or not str(v).strip():
        return None
    return float(v)


# If set (e.g. 99.99), each 2D plane is clipped to np.percentile(plane, value)
# before Cellpose eval for runs under .../488nm_crop/ only.
CELLPOSE_PRE_CLIP_PCTILE = _optional_float_env("CELLPOSE_PRE_CLIP_PCTILE")
```

## 2. `segmentation_multiscale_cellpose_3D/multiscale_cellpose/segmentation_cellpose_2d.py`

- In `from pipeline_config import (...)`, add `CELLPOSE_PRE_CLIP_PCTILE`, `OUTPUT_DIR`.
- Add `from pathlib import Path` if not already there (Path is used in file — already `from pathlib import Path` at line 14).

Replace the block from `img = np.asarray(img, dtype=np.float32)` through `model.eval(` with:

```python
        img = np.asarray(img, dtype=np.float32)

        pct = CELLPOSE_PRE_CLIP_PCTILE
        out_dir_s = Path(OUTPUT_DIR).resolve().as_posix()
        if pct is not None and "488nm_crop" in out_dir_s:
            hi = float(np.percentile(img, pct))
            img_eval = np.minimum(img, hi).astype(np.float32)
        else:
            img_eval = img

        for d in diameters:
            out_path = os.path.join(dm_dir, f"{stem}_diameter_{d}.tif")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                continue
            masks, _, _ = model.eval(
                img_eval,
                diameter=d,
                channels=None,
                normalize=True,
                flow_threshold=CELLPOSE_FLOW_THRESHOLD,
                cellprob_threshold=CELLPOSE_CELLPROB_THRESHOLD,
            )
```

## 3. Submit one GPU job (Sample10_Position0, 4_24_25)

From `segmentation_multiscale_cellpose_3D/`:

```bash
export CELLPOSE_PRE_CLIP_PCTILE=99.99
OUT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_multiscale_cellpose_3D/output/4_24_25_CGN_6_10_2/Sample10_Position0_decon_dsr/488nm_crop"
rm -rf "$OUT"

cd /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_multiscale_cellpose_3D
export CMAP_INPUT_DIR="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysisScratch/Gutierrez/CMAP_cropped_copies/4_24_25_CGN_6_10_2/Sample10_Position0_decon_dsr"
export CMAP_OUTPUT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/segmentation_multiscale_cellpose_3D/output"

bsub -q rhel88_gpu -J clip488_Pos0 -n 1 -R "rusage[mem=16000] span[hosts=1]" -W 2880 \
  -gpu "num=1:j_exclusive=yes" -env "all" \
  bash scripts/bsub_cmap_sample_full.sh
```

`-env all` forwards `CELLPOSE_PRE_CLIP_PCTILE` from the submission shell into the job.

## 4. After job: check 488 indexed max

```bash
python3 -c "import numpy as np, tifffile as tf; p='.../488nm_crop/segmentation_3D_masks/488nm_crop_3D_indexed.tif'; a=tf.memmap(p,mode='r'); print(int(a.max()))"
```

(Use full path to indexed tif.)
