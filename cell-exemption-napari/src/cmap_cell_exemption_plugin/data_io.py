"""Filesystem, TIFF, coordinate CSV, and annotation CSV helpers."""

from __future__ import annotations

import csv
import io
import os
import re
import tempfile
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import pandas as pd
import tifffile

from .models import (
    ANNOTATION_CSV_GLOB,
    ANNOTATION_CSV_PREFIX,
    CELL_BOX_SUBDIR,
    COMBINED_TIFF_NAME,
    COORDINATE_CSV_NAME,
    AnnotationRow,
    CellBoxRecord,
    SampleRecord,
    dataframe_to_rows,
    rows_to_dataframe,
)

REQUIRED_COORD_COLUMNS = ("cell_id", "z0", "z1", "y0", "y1", "x0", "x1")


def path_key(path: Path | str) -> str:
    """Mount-stable absolute path key for matching annotations.

    This intentionally does not call `resolve()` or `realpath()` because CMAP
    data may be reachable through multiple mount aliases.
    """

    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    return os.path.normpath(str(p))


def infer_batch_sample(sample_dir: Path, root: Path) -> tuple[str, str]:
    """Infer `(batch, sample)` from a sample directory relative to the root."""

    try:
        rel = sample_dir.relative_to(root)
        parts = rel.parts
    except ValueError:
        parts = sample_dir.parts
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    if len(parts) == 1:
        return "", parts[0]
    return "", sample_dir.name


def sample_record_from_dir(sample_dir: Path, root: Path | None = None) -> SampleRecord | None:
    """Return a sample record if the directory has the expected CMAP files."""

    image_path = sample_dir / COMBINED_TIFF_NAME
    coordinate_csv = sample_dir / CELL_BOX_SUBDIR / COORDINATE_CSV_NAME
    if not image_path.is_file() or not coordinate_csv.is_file():
        return None
    root_for_rel = root or sample_dir.parent
    batch, sample = infer_batch_sample(sample_dir, root_for_rel)
    return SampleRecord(
        batch=batch,
        sample=sample,
        sample_dir=sample_dir,
        image_path=image_path,
        coordinate_csv=coordinate_csv,
    )


def discover_samples(root: Path) -> list[SampleRecord]:
    """Find reconstructed sample volumes under an output root.

    The walk only checks file names and does not open TIFF payloads.
    """

    root = Path(root).expanduser()
    if not root.is_dir():
        return []

    direct = sample_record_from_dir(root, root.parent)
    if direct is not None:
        return [direct]

    samples: list[SampleRecord] = []
    for dirpath, _dirnames, filenames in os.walk(os.fspath(root), followlinks=True):
        if COMBINED_TIFF_NAME not in filenames:
            continue
        sample_dir = Path(dirpath)
        rec = sample_record_from_dir(sample_dir, root)
        if rec is not None:
            samples.append(rec)
    samples.sort(key=lambda r: (r.batch.lower(), r.sample.lower(), path_key(r.sample_dir)))
    return samples


def load_cell_boxes(path: Path) -> list[CellBoxRecord]:
    """Load `cell_coordinates.csv` rows as sorted cell box records."""

    rows: list[CellBoxRecord] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COORD_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(missing)}")
        for line_no, row in enumerate(reader, start=2):
            try:
                rows.append(
                    CellBoxRecord(
                        cell_id=int(row["cell_id"]),
                        z0=int(row["z0"]),
                        z1=int(row["z1"]),
                        y0=int(row["y0"]),
                        y1=int(row["y1"]),
                        x0=int(row["x0"]),
                        x1=int(row["x1"]),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: malformed coordinate row") from exc
    rows.sort(key=lambda r: r.cell_id)
    return rows


class _InMemoryTiffPages:
    """Hold a compressed TIFF in RAM and decode pages on demand.

    The CMAP combined volumes are ~2 GB float32 stacks compressed with DEFLATE
    and stored on networked storage. Two things make a naive
    ``tifffile.imread`` slow (~16 s/image):

    * walking the 345 per-page IFDs triggers hundreds of random NFS seeks
      (~9 s cold), and
    * eagerly decompressing every page costs another ~7 s even though the
      viewer only ever shows a handful of slices at a time.

    Reading the whole file sequentially into memory is ~2 s, after which IFD
    parsing is instant (no network seeks) and individual pages decode in
    ~0.03 s. We therefore slurp the bytes once and decode pages lazily.
    """

    # Cap on decoded planes kept in RAM (each ~12.5 MB float32). Decoding a
    # DEFLATE page costs ~0.03 s, so caching makes z-scrolling revisits instant
    # instead of re-decoding every slice change.
    _PAGE_CACHE_MAX = 128

    def __init__(self, path: str):
        with open(path, "rb") as fh:
            self._raw = fh.read()
        self._tif = tifffile.TiffFile(io.BytesIO(self._raw))
        self._lock = threading.Lock()
        self._page_cache: "OrderedDict[int, object]" = OrderedDict()
        series = self._tif.series[0]
        self.shape = tuple(series.shape)
        self.dtype = series.dtype

    def read_page(self, page_index: int):
        # tifffile's page reader seeks within the shared in-memory handle, so
        # serialize access for dask's threaded scheduler. Cached planes return
        # immediately so repeated z navigation does not re-decode.
        with self._lock:
            cached = self._page_cache.get(page_index)
            if cached is not None:
                self._page_cache.move_to_end(page_index)
                return cached
            plane = self._tif.pages[page_index].asarray()
            self._page_cache[page_index] = plane
            self._page_cache.move_to_end(page_index)
            while len(self._page_cache) > self._PAGE_CACHE_MAX:
                self._page_cache.popitem(last=False)
            return plane

    def read_full(self):
        with self._lock:
            return self._tif.asarray()


# Keep a tiny LRU of in-RAM readers so the lazy dask graph for the current
# (and previous) image can still decode pages, without letting 2 GB buffers
# accumulate as the reviewer walks through dozens of samples.
_TIFF_READER_CACHE: "OrderedDict[str, _InMemoryTiffPages]" = OrderedDict()
_TIFF_READER_CACHE_LOCK = threading.Lock()
_TIFF_READER_CACHE_MAX = 2


def _register_reader(key: str, reader: _InMemoryTiffPages) -> None:
    with _TIFF_READER_CACHE_LOCK:
        _TIFF_READER_CACHE[key] = reader
        _TIFF_READER_CACHE.move_to_end(key)
        while len(_TIFF_READER_CACHE) > _TIFF_READER_CACHE_MAX:
            _TIFF_READER_CACHE.popitem(last=False)


def _read_cached_page(key: str, page_index: int):
    with _TIFF_READER_CACHE_LOCK:
        reader = _TIFF_READER_CACHE.get(key)
    if reader is None:
        raise KeyError(f"TIFF reader for {key} is no longer cached")
    return reader.read_page(page_index)


def load_combined_tiff(path: Path):
    """Lazily read a reconstructed combined TIFF.

    Returns a dask array of shape ``(Z, C, Y, X)`` whose pages decode on demand
    so navigation stays fast on networked storage. Falls back to an eager numpy
    read when dask is unavailable or the layout is unexpected.
    """

    key = path_key(path)
    reader = _InMemoryTiffPages(str(path))
    _register_reader(key, reader)

    try:
        import dask
        import dask.array as da
    except Exception:
        return reader.read_full()

    shape = reader.shape
    if len(shape) != 4:
        return reader.read_full()

    z_size, c_size, y_size, x_size = shape
    dtype = reader.dtype
    planes = []
    for z in range(z_size):
        channels = [
            da.from_delayed(
                dask.delayed(_read_cached_page)(key, z * c_size + c),
                shape=(y_size, x_size),
                dtype=dtype,
            )
            for c in range(c_size)
        ]
        planes.append(da.stack(channels, axis=0))
    return da.stack(planes, axis=0)


def sanitize_filename_token(text: str | None) -> str:
    """Make a string safe to embed in a filename (alnum, `_`, `-`, `.`)."""

    text = (text or "").strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.-]", "", text)
    return text.strip("_.-")


def annotation_csv_path(
    output_dir: Path, stamp: str | None = None, reviewer: str | None = None
) -> Path:
    """Return the current session annotation CSV path.

    When a ``reviewer`` is given, it is sanitized and embedded in the filename
    so each reviewer's output is easy to identify, e.g.
    ``cell_exemption_annotations_<reviewer>_<stamp>.csv``.
    """

    if stamp is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rev = sanitize_filename_token(reviewer)
    if rev:
        name = f"{ANNOTATION_CSV_PREFIX}_{rev}_{stamp}.csv"
    else:
        name = f"{ANNOTATION_CSV_PREFIX}_{stamp}.csv"
    return Path(output_dir) / name


def load_annotation_csv(path: Path) -> list[AnnotationRow]:
    """Load one annotation CSV, tolerating missing optional columns."""

    if not path.is_file():
        return []
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return dataframe_to_rows(df)


def load_annotation_folder(output_dir: Path) -> dict[tuple[str, str, str], AnnotationRow]:
    """Load all prior annotation CSVs oldest-first so newer rows override older rows."""

    output_dir = Path(output_dir)
    rows_by_key: dict[tuple[str, str, str], AnnotationRow] = {}
    if not output_dir.is_dir():
        return rows_by_key
    candidates = sorted(output_dir.glob(ANNOTATION_CSV_GLOB), key=lambda p: p.stat().st_mtime)
    for csv_path in candidates:
        for row in load_annotation_csv(csv_path):
            rows_by_key[row.key] = row
    return rows_by_key


def save_annotations_atomic(path: Path, rows: list[AnnotationRow]) -> None:
    """Atomically write annotation rows to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows_to_dataframe(rows)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".annotations_")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False, lineterminator="\n")
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
