"""DINOv2 ViT-B/14 (2.5D) cell-embedding subpackage.

Public surface:

- :class:`DinoV2Config` — frozen dataclass of defaults / CLI overrides.
- :class:`DinoV2Encoder` — frozen torch.hub-loaded encoder wrapper.
- :func:`extract_one_cell` / :func:`extract_sample` — high-level pipeline
  entrypoints used by ``features/extract_dinov2_embeddings.py``.

The active per-cell input contract is the (Z, 5, Y, X) cell crops written by
``features/crop_cells.py`` (``cell_boxing/cell_<NNNN>.tif``) with channels
``[642, 488, 560, Primary_Cell_Mask, Mask_Boundary]``.  Older 4-channel and
raw 3-channel crops are tolerated.
"""
from __future__ import annotations

from .aggregate import mean_pool
from .config import DinoV2Config
from .encoder import DinoV2Encoder
from .pipeline import extract_one_cell, extract_sample

__all__ = [
    "DinoV2Config",
    "DinoV2Encoder",
    "mean_pool",
    "extract_one_cell",
    "extract_sample",
]
