"""Per-sample volume intensity bounds: compute once from TIFF, reuse via CSV.

Reading a small CSV avoids reloading ``filtered_642_combined.tif`` for every
DINOv2 extraction or hyperparameter sweep.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import tifffile

logger = logging.getLogger(__name__)

# Default filename next to per-sample pipeline outputs (same folder as combined).
DEFAULT_BOUNDS_CSV_NAME = "dinov2_volume_norm_bounds.csv"


def default_bounds_csv_path(output_dir: Path, filename: str | None = None) -> Path:
    """Return the per-sample bounds CSV path.

    ``filename`` lets callers pick the per-variant bounds CSV name from
    ``postprocess.VARIANT_FILES``; when omitted, the legacy filename
    (``dinov2_volume_norm_bounds.csv``) is used so existing 642 callers stay
    byte-identical.
    """
    return Path(output_dir) / (filename or DEFAULT_BOUNDS_CSV_NAME)


def _bounds_from_zcyx_volume(
    vol: np.ndarray,
    *,
    p_high: float,
    max_voxels_per_channel: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``vol`` is (Z, C, Y, X) with C >= 3; may be ndarray or numpy memmap."""
    if vol.ndim != 4 or vol.shape[1] < 3:
        raise ValueError(f"Expected (Z, C>=3, Y, X) combined TIFF, got shape {vol.shape}")

    z, _, y, x = int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2]), int(vol.shape[3])
    total_vox = z * y * x
    lo = np.zeros(3, dtype=np.float32)

    zstep = max(1, min(64, z))
    for c in range(3):
        mch = np.finfo(np.float32).max
        for zs in range(0, z, zstep):
            ze = min(z, zs + zstep)
            slab = np.asarray(vol[zs:ze, c, :, :], dtype=np.float32)
            mch = min(mch, float(slab.min()))
        lo[c] = mch

    rng = np.random.default_rng(seed)
    n_samp = min(max_voxels_per_channel, total_vox)
    zz = rng.integers(0, z, size=n_samp, dtype=np.int64)
    yy = rng.integers(0, y, size=n_samp, dtype=np.int64)
    xx = rng.integers(0, x, size=n_samp, dtype=np.int64)

    hi = np.zeros(3, dtype=np.float32)
    for c in range(3):
        samp = np.asarray(vol[zz, c, yy, xx], dtype=np.float32)
        hi[c] = float(np.percentile(samp, p_high))

    for c in range(3):
        if hi[c] <= lo[c]:
            hi[c] = lo[c] + 1e-6
    return lo, hi


def compute_bounds_from_combined_tif(
    combined_path: Path,
    *,
    p_high: float = 99.99,
    max_voxels_per_channel: int = 5_000_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Min and ``p_high`` percentile per intensity channel (first 3 of ZCYX).

    Uses memory-mapping when the TIFF layout allows it; otherwise loads the
    full volume (typical for zlib-compressed ImageJ OME TIFFs — needs enough RAM).

    For large volumes, the upper percentile is estimated from a random voxel
    subsample per channel.

    Returns:
        ``lo, hi`` each float32 ``(3,)`` — per-channel minimum and ``p_high``
        percentile (approximate when subsampling).
    """
    path = str(combined_path)
    try:
        vol = tifffile.memmap(path, mode="r")
    except ValueError as exc:
        if "mappable" not in str(exc).lower():
            raise
        nbytes = 0
        try:
            with tifffile.TiffFile(path) as tf:
                shp = tf.series[0].shape
                dt = tf.series[0].dtype
                if len(shp) == 4:
                    nbytes = int(np.prod(shp)) * np.dtype(dt).itemsize
        except Exception:
            pass
        logger.warning(
            "TIFF is not memory-mappable (e.g. zlib). Loading full volume%s.",
            f" (~{nbytes / 1e9:.2f} GiB)" if nbytes else "",
        )
        vol = tifffile.imread(path)
    if vol.ndim != 4 or vol.shape[1] < 3:
        raise ValueError(f"Expected (Z, C>=3, Y, X) combined TIFF, got shape {vol.shape}")
    return _bounds_from_zcyx_volume(
        vol,
        p_high=p_high,
        max_voxels_per_channel=max_voxels_per_channel,
        seed=seed,
    )


def write_bounds_csv(path: Path, lo: np.ndarray, hi: np.ndarray, *, source_tif: str) -> None:
    """Write ``channel,lo,hi`` rows plus metadata comment in header row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# source_tif", source_tif])
        w.writerow(["channel_index", "lo", "hi"])
        for i in range(3):
            w.writerow([i, f"{float(lo[i]):.8g}", f"{float(hi[i]):.8g}"])
    Path(tmp).replace(path)


def read_bounds_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(lo, hi)`` float32 ``(3,)`` from CSV written by :func:`write_bounds_csv`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    lo_list: list[float] = []
    hi_list: list[float] = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#"):
                continue
            if row[0] == "channel_index":
                continue
            lo_list.append(float(row[1]))
            hi_list.append(float(row[2]))
    if len(lo_list) != 3 or len(hi_list) != 3:
        raise ValueError(f"Expected 3 channel rows in {path}, got {len(lo_list)}")
    lo = np.array(lo_list, dtype=np.float32)
    hi = np.array(hi_list, dtype=np.float32)
    return lo, hi
