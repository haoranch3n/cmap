#!/usr/bin/env python3
"""Create a tiny synthetic sample and run DINOv2 on one cell (orthogonal + volume norm).

Uses ``--recompute-volume-norm`` once to build the bounds CSV from the combined
TIFF, then encodes one ``cell_*.tif``. GPU is used when ``torch.cuda.is_available()``.

Run from repo root::

    python scripts/smoke_dinov2_one_cell.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.dinov2 import DinoV2Config, DinoV2Encoder, extract_sample  # noqa: E402


def _synthetic_cell_crop(z: int = 8, y: int = 40, x: int = 40, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    intensities = rng.uniform(50, 1500, size=(z, 3, y, x)).astype(np.float32)
    intensities[0] = 0.0
    mask = np.zeros((z, y, x), dtype=np.float32)
    cy, cx = y // 2, x // 2
    half = min(y, x) // 4
    mask[:, cy - half : cy + half, cx - half : cx + half] = 1.0
    boundary = np.zeros_like(mask)
    boundary[:, cy - half, cx - half : cx + half] = 1.0
    return np.concatenate(
        [intensities, mask[:, None], boundary[:, None]],
        axis=1,
    ).astype(np.float32)


def main() -> int:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tmp = Path(tempfile.mkdtemp(prefix="dinov2_smoke_"))
    print(f"temp sample dir: {tmp}")

    zc, yc, xc = 10, 48, 48
    rng = np.random.default_rng(0)
    combined = rng.uniform(0, 2000, size=(zc, 4, yc, xc)).astype(np.float32)
    tifffile.imwrite(
        str(tmp / "filtered_642_combined.tif"),
        combined,
        imagej=True,
        metadata={"axes": "ZCYX"},
    )

    box = tmp / "cell_boxing"
    box.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(box / "cell_0001.tif"),
        _synthetic_cell_crop(),
        imagej=True,
        metadata={"axes": "ZCYX"},
    )

    cfg = DinoV2Config(
        target_size=224,
        batch_size=8,
        device=device,
        extraction_mode="orthogonal_concat",
        norm_scope="volume",
        empty_slice_nonzero_frac=0.0,
        apply_mask=False,
    )
    encoder = DinoV2Encoder(cfg)
    rc = extract_sample(
        tmp,
        cfg,
        encoder=encoder,
        force=True,
        recompute_volume_norm=True,
        max_cells=1,
    )
    if rc != 0:
        return rc

    emb = np.load(str(tmp / "cell_qc" / "dinov2_embeddings.npy"))
    print(f"dinov2_embeddings.npy shape: {emb.shape}  dtype: {emb.dtype}")
    print(f"  first row L2 norm: {float(np.linalg.norm(emb[0])):.4f}")
    bounds_csv = tmp / "dinov2_volume_norm_bounds.csv"
    print(f"bounds CSV exists: {bounds_csv.is_file()}")
    if bounds_csv.is_file():
        print(bounds_csv.read_text()[:500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
