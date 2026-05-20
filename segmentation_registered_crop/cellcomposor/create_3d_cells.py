"""
Assemble a 3D cell label volume from stacked 2D masks, applying internal
quality filters (gap-bridging, z-span, volume, area-consistency).

Adapted from segmentation/assemble3d/create_3d_cells.py for use with
per-crop per-channel directory layouts.  Postprocessing steps (intensity
filtering, mask combination, quantification) are intentionally excluded.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from skimage.io import imread
from skimage.measure import label as sklabel
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from tifffile import imwrite

_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

# Add the assemble3d dir of the base segmentation module so we can reuse
# the 2D→3D matching code (match_2d_cells.py) without copying it.
_BASE_ASSEMBLE3D = (
    Path(__file__).resolve().parents[2] / "segmentation" / "assemble3d"
)
if str(_BASE_ASSEMBLE3D) not in sys.path:
    sys.path.insert(0, str(_BASE_ASSEMBLE3D))

from segmentation_3d.match_2d_cells import matching_cells_2D  # noqa: E402

from pipeline_config import (  # noqa: E402
    JI_THRESHOLD,
    MAX_AREA_CHANGE_RATIO,
    MIN_CELL_VOLUME_3D,
    MIN_CELL_Z_SPAN,
)


# ---------------------------------------------------------------------------
# Post-assembly quality filters
# ---------------------------------------------------------------------------

def filter_short_z_cells(seg_3d: np.ndarray, min_z_span: int = MIN_CELL_Z_SPAN) -> np.ndarray:
    seg_out = seg_3d.copy()
    props = regionprops(seg_3d)
    n_removed = 0
    for prop in tqdm.tqdm(props, desc=f"Z-span filter (< {min_z_span})", leave=False):
        zmin, _, _, zmax, _, _ = prop.bbox
        if zmax - zmin < min_z_span:
            seg_out[seg_3d == prop.label] = 0
            n_removed += 1
    if n_removed:
        print(f"  Z-span filter: removed {n_removed} cell(s)")
    return seg_out


def make_color_mask(seg_3d: np.ndarray) -> np.ndarray:
    labels = np.unique(seg_3d)
    labels = labels[labels != 0]
    if labels.size == 0:
        return np.zeros(seg_3d.shape + (3,), dtype=np.uint8)
    cmap = plt.get_cmap("tab20")
    rng = np.random.default_rng()
    colors = (np.array([cmap(rng.integers(0, 20))[:3] for _ in labels]) * 255).astype(np.uint8)
    max_lbl = int(seg_3d.max())
    lut = np.zeros((max_lbl + 1, 3), dtype=np.uint8)
    lut[labels] = colors
    color_mask = lut[seg_3d].copy()
    boundaries = find_boundaries(seg_3d, mode="thick", connectivity=1)
    color_mask[boundaries] = (0, 0, 0)
    return color_mask


def bridge_gaps(seg_3d: np.ndarray, ji_thre: float = 0.3) -> np.ndarray:
    props = regionprops(seg_3d)
    if not props:
        return seg_3d
    cell_zmin: dict[int, int] = {}
    cell_zmax: dict[int, int] = {}
    for p in props:
        zmin, _, _, zmax, _, _ = p.bbox
        cell_zmin[p.label] = zmin
        cell_zmax[p.label] = zmax

    zmin_to_labels: dict[int, list[int]] = {}
    for lbl, zmin in cell_zmin.items():
        zmin_to_labels.setdefault(zmin, []).append(lbl)

    merge_map: dict[int, int] = {}
    fill_slices: list[tuple[int, int, np.ndarray]] = []

    for label_a in sorted(cell_zmax, key=cell_zmax.get):
        if label_a in merge_map:
            continue
        zmax_a = cell_zmax[label_a]
        last_z_a = zmax_a - 1
        z_gap = zmax_a
        z_check = zmax_a + 1
        if z_check >= seg_3d.shape[0]:
            continue
        candidates = zmin_to_labels.get(z_check, [])
        if not candidates:
            continue
        mask_a = seg_3d[last_z_a] == label_a
        best_ji, best_b = 0.0, None
        for label_b in candidates:
            if label_b in merge_map:
                continue
            mask_b = seg_3d[z_check] == label_b
            inter = int(np.sum(mask_a & mask_b))
            union = int(np.sum(mask_a | mask_b))
            if union > 0:
                ji = inter / union
                if ji > best_ji and ji > ji_thre:
                    best_ji = ji
                    best_b = label_b
        if best_b is not None:
            merge_map[best_b] = label_a
            mask_b = seg_3d[z_check] == best_b
            fill_mask = mask_a | mask_b
            fill_slices.append((z_gap, label_a, fill_mask))

    if not merge_map:
        return seg_3d

    max_label = int(seg_3d.max())
    lut = np.arange(max_label + 1, dtype=np.int32)
    for later, earlier in merge_map.items():
        if later <= max_label:
            lut[later] = earlier
    seg_3d = lut[seg_3d.astype(np.int32)]
    for z, label_val, fill_mask in fill_slices:
        bg = seg_3d[z] == 0
        seg_3d[z][fill_mask & bg] = label_val
    print(
        f"  Gap bridging: merged {len(merge_map)} fragment(s), "
        f"filled {len(fill_slices)} missing slice(s)"
    )
    return seg_3d


def split_disconnected_3d(seg_3d: np.ndarray) -> np.ndarray:
    props = regionprops(seg_3d)
    next_label = int(seg_3d.max()) + 1
    n_split = 0
    for p in tqdm.tqdm(props, desc="3D connectivity split", leave=False):
        zmin, ymin, xmin, zmax, ymax, xmax = p.bbox
        cell_mask = seg_3d[zmin:zmax, ymin:ymax, xmin:xmax] == p.label
        cc, n_cc = sklabel(cell_mask, return_num=True)
        if n_cc <= 1:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        largest = int(np.argmax(sizes)) + 1
        sub = seg_3d[zmin:zmax, ymin:ymax, xmin:xmax]
        for cc_id in range(1, n_cc + 1):
            if cc_id == largest:
                continue
            sub[cc == cc_id] = next_label
            next_label += 1
            n_split += 1
    if n_split:
        print(f"  Split {n_split} disconnected component(s)")
    return seg_3d


def filter_size_inconsistent(seg_3d: np.ndarray, max_ratio: float = MAX_AREA_CHANGE_RATIO) -> np.ndarray:
    props = regionprops(seg_3d)
    n_trimmed = 0
    n_removed = 0
    for p in props:
        zmin, _, _, zmax, _, _ = p.bbox
        if zmax - zmin <= 1:
            continue
        z_areas = []
        for z in range(zmin, zmax):
            area = int(np.sum(seg_3d[z] == p.label))
            if area > 0:
                z_areas.append((z, area))
        if len(z_areas) < 2:
            continue
        while len(z_areas) >= 2:
            lo, hi = sorted((z_areas[-1][1], z_areas[-2][1]))
            if lo > 0 and hi / lo > max_ratio:
                bad_z = z_areas.pop()[0]
                seg_3d[bad_z][seg_3d[bad_z] == p.label] = 0
                n_trimmed += 1
            else:
                break
        while len(z_areas) >= 2:
            lo, hi = sorted((z_areas[0][1], z_areas[1][1]))
            if lo > 0 and hi / lo > max_ratio:
                bad_z = z_areas.pop(0)[0]
                seg_3d[bad_z][seg_3d[bad_z] == p.label] = 0
                n_trimmed += 1
            else:
                break
        if len(z_areas) < 2:
            for z, _ in z_areas:
                seg_3d[z][seg_3d[z] == p.label] = 0
            n_removed += 1
    parts = []
    if n_trimmed:
        parts.append(f"trimmed {n_trimmed} slice(s)")
    if n_removed:
        parts.append(f"removed {n_removed} cell(s)")
    if parts:
        print(f"  Size-consistency: {', '.join(parts)}")
    return seg_3d


def filter_small_volumes(seg_3d: np.ndarray, min_volume: int = MIN_CELL_VOLUME_3D) -> np.ndarray:
    if min_volume <= 0:
        return seg_3d
    counts = np.bincount(seg_3d.ravel())
    small = np.zeros(len(counts), dtype=bool)
    small[1:] = counts[1:] < min_volume
    if small.any():
        lut = np.arange(len(counts), dtype=np.int32)
        lut[small] = 0
        seg_3d = lut[seg_3d.astype(np.int32)]
        print(f"  Volume filter: removed {int(small.sum())} cell(s)")
    return seg_3d


def absorb_short_fragments(
    seg_3d: np.ndarray,
    max_short_span: int = 15,
    ji_thre: float = 0.3,
    max_iterations: int = 5,
) -> np.ndarray:
    total_merged = 0
    for _ in range(max_iterations):
        props = regionprops(seg_3d)
        if not props:
            break
        cell_spans: dict[int, tuple[int, int]] = {}
        for p in props:
            zmin, _, _, zmax, _, _ = p.bbox
            cell_spans[p.label] = (int(zmin), int(zmax))

        merge_map: dict[int, int] = {}
        short_cells = sorted(
            [p for p in props if cell_spans[p.label][1] - cell_spans[p.label][0] <= max_short_span],
            key=lambda p: cell_spans[p.label][1] - cell_spans[p.label][0],
        )

        for p in short_cells:
            lbl = p.label
            if lbl in merge_map:
                continue
            zmin, zmax = cell_spans[lbl]
            first_mask = seg_3d[zmin] == lbl
            last_mask = seg_3d[zmax - 1] == lbl
            best_target, best_ji = None, 0.0

            for z in range(max(0, zmin - 3), zmin):
                ref = first_mask
                for o in np.unique(seg_3d[z][ref]):
                    o = int(o)
                    if o == 0 or o == lbl or o in merge_map:
                        continue
                    o_span = cell_spans.get(o)
                    if o_span is None or o_span[1] - o_span[0] <= zmax - zmin:
                        continue
                    o_mask = seg_3d[z] == o
                    inter = int(np.sum(ref & o_mask))
                    union = int(np.sum(ref | o_mask))
                    if union > 0:
                        ji = inter / union
                        if ji > best_ji and ji > ji_thre:
                            best_ji, best_target = ji, o

            for z in range(zmax, min(seg_3d.shape[0], zmax + 3)):
                ref = last_mask
                for o in np.unique(seg_3d[z][ref]):
                    o = int(o)
                    if o == 0 or o == lbl or o in merge_map:
                        continue
                    o_span = cell_spans.get(o)
                    if o_span is None or o_span[1] - o_span[0] <= zmax - zmin:
                        continue
                    o_mask = seg_3d[z] == o
                    inter = int(np.sum(ref & o_mask))
                    union = int(np.sum(ref | o_mask))
                    if union > 0:
                        ji = inter / union
                        if ji > best_ji and ji > ji_thre:
                            best_ji, best_target = ji, o

            if best_target is not None:
                merge_map[lbl] = best_target

        if not merge_map:
            break
        max_label = int(seg_3d.max())
        lut = np.arange(max_label + 1, dtype=np.int32)
        for short_lbl, long_lbl in merge_map.items():
            lut[short_lbl] = long_lbl
        seg_3d = lut[seg_3d.astype(np.int32)]
        total_merged += len(merge_map)

    if total_merged:
        print(f"  Fragment absorption: merged {total_merged} short fragment(s)")
    return seg_3d


def relabel_contiguous(seg_3d: np.ndarray) -> np.ndarray:
    labels = np.unique(seg_3d)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return seg_3d
    max_label = int(labels.max())
    lut = np.zeros(max_label + 1, dtype=np.int32)
    for new_lbl, old_lbl in enumerate(labels, start=1):
        lut[int(old_lbl)] = new_lbl
    return lut[seg_3d.astype(np.int32)]


def create_3d_cells(
    stacked_tif: Path,
    seg3d_dir: Path,
    ji_thre: float = JI_THRESHOLD,
    force: bool = False,
) -> str | None:
    """
    Build a 3D indexed label volume from a stacked 2D mask TIFF.

    Args:
        stacked_tif: Path to the ``*_2D_stacked.tif`` file.
        seg3d_dir:   Output directory for the indexed 3D mask.
        ji_thre:     Jaccard index threshold for cell matching across Z.
        force:       Re-run even if the indexed mask already exists.

    Returns:
        Path to the indexed mask TIFF, or None on failure.
    """
    stacked_tif = Path(stacked_tif)
    seg3d_dir = Path(seg3d_dir)
    seg3d_dir.mkdir(parents=True, exist_ok=True)

    stem = stacked_tif.stem
    if stem.endswith("_2D_stacked"):
        stem = stem[: -len("_2D_stacked")]

    index_mask_path = str(seg3d_dir / f"{stem}_3D_indexed.tif")

    if not force and os.path.exists(index_mask_path) and os.path.getsize(index_mask_path) > 0:
        return index_mask_path

    print(f"  3D assembly: {stacked_tif.name}")
    seg_2d_stack = imread(str(stacked_tif))
    seg_3d = matching_cells_2D(seg_2d_stack, JI_thre=ji_thre)

    seg_3d = bridge_gaps(seg_3d, ji_thre=ji_thre)
    seg_3d = absorb_short_fragments(seg_3d, max_short_span=15, ji_thre=ji_thre)
    seg_3d = filter_short_z_cells(seg_3d, min_z_span=MIN_CELL_Z_SPAN)
    seg_3d = split_disconnected_3d(seg_3d)
    seg_3d = filter_size_inconsistent(seg_3d, max_ratio=MAX_AREA_CHANGE_RATIO)
    seg_3d = filter_small_volumes(seg_3d, min_volume=MIN_CELL_VOLUME_3D)
    seg_3d = filter_short_z_cells(seg_3d, min_z_span=MIN_CELL_Z_SPAN)
    seg_3d = relabel_contiguous(seg_3d)

    n_cells = int(seg_3d.max())
    print(f"  3D assembly complete: {n_cells} cell(s) after filtering")

    tmp = index_mask_path + ".tmp"
    imwrite(tmp, seg_3d.astype(np.uint16))
    os.replace(tmp, index_mask_path)
    return index_mask_path
