"""
2D Cellpose segmentation for DAPI planes using the default Cellpose 4 pretrained model (cpsam).

Reads `*_DAPI.tif` under `output/tif_planes`, writes one label mask per plane under
`output/segmentation_2D_planes/.../<plane_stem>/<plane_stem>_final_mask.tif`
so downstream 3D stacking scripts can consume the same layout as before.
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.io import imread
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline_config import (  # noqa: E402
    CELLPOSE_PRETRAINED_MODEL,
    SEGMENTATION_2D_DIR,
    TIF_PLANES_DIR,
)

try:
    import torch

    def gpu_available():
        return torch.cuda.is_available()
except Exception:
    def gpu_available():
        return False

from cellpose import models


def discover_dapi_tifs(tif_planes_root: Path):
    pattern = str(tif_planes_root / "**" / "*_DAPI.tif")
    return sorted(glob.glob(pattern, recursive=True))


def plane_stem_from_dapi_path(dapi_path: str) -> str:
    base = os.path.basename(dapi_path)
    if base.endswith("_DAPI.tif"):
        return base[: -len("_DAPI.tif")]
    return Path(base).stem


def output_final_mask_path(dapi_path: str, tif_planes_root: Path, seg_root: Path) -> str:
    rel_parent = os.path.relpath(os.path.dirname(dapi_path), tif_planes_root)
    stem = plane_stem_from_dapi_path(dapi_path)
    out_dir = os.path.join(seg_root, rel_parent, stem)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{stem}_final_mask.tif")


def run_segmentation(
    tif_planes_root: Path | None = None,
    seg_root: Path | None = None,
    pretrained_model: str | None = None,
    gpu: bool | None = None,
):
    tif_planes_root = Path(tif_planes_root or TIF_PLANES_DIR)
    seg_root = Path(seg_root or SEGMENTATION_2D_DIR)
    pretrained_model = pretrained_model or CELLPOSE_PRETRAINED_MODEL
    if gpu is None:
        gpu = gpu_available()

    dapi_files = discover_dapi_tifs(tif_planes_root)
    if not dapi_files:
        print(f"No *_DAPI.tif files under {tif_planes_root}")
        return

    print(f"Using Cellpose pretrained_model={pretrained_model!r}, gpu={gpu}")
    print(f"Found {len(dapi_files)} DAPI plane(s).")

    model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)

    for dapi_path in tqdm(dapi_files, desc="Cellpose 2D"):
        out_mask = output_final_mask_path(
            dapi_path, os.fspath(tif_planes_root), os.fspath(seg_root)
        )
        if os.path.exists(out_mask) and os.path.getsize(out_mask) > 0:
            continue

        img = imread(dapi_path)
        if img.ndim == 3:
            img = img.squeeze()
        if img.ndim != 2:
            raise ValueError(f"Expected 2D plane, got shape {img.shape} for {dapi_path}")

        img = np.asarray(img, dtype=np.float32)
        masks, _flows, _styles = model.eval(
            img,
            diameter=None,
            channels=None,
            normalize=True,
        )

        masks = np.asarray(masks).astype(np.uint16)
        tifffile.imwrite(out_mask, masks)


if __name__ == "__main__":
    run_segmentation()
