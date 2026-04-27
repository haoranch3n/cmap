from __future__ import annotations

import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from skimage.io import imread
from skimage.measure import label as sklabel
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from tifffile import imwrite

try:
    from segmentation_3d.match_2d_cells import matching_cells_2D
except ModuleNotFoundError:
    from segmentation.assemble3d.segmentation_3d.match_2d_cells import matching_cells_2D

_SEG_ROOT = Path(__file__).resolve().parents[1]
if str(_SEG_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEG_ROOT))

try:
    from config import (
        JI_THRESHOLD,
        MAX_AREA_CHANGE_RATIO,
        MIN_CELL_VOLUME_3D,
        MIN_CELL_Z_SPAN,
        SEGMENTATION_2D_STACKED_DIR,
        SEGMENTATION_3D_DIR,
    )
except ModuleNotFoundError:
    from segmentation.config import (
        JI_THRESHOLD,
        MAX_AREA_CHANGE_RATIO,
        MIN_CELL_VOLUME_3D,
        MIN_CELL_Z_SPAN,
        SEGMENTATION_2D_STACKED_DIR,
        SEGMENTATION_3D_DIR,
    )

BASE_DIR = os.fspath(SEGMENTATION_2D_STACKED_DIR)


def filter_short_z_cells(seg_3d, min_z_span=MIN_CELL_Z_SPAN):
    seg_3d_filtered = seg_3d.copy()
    props = regionprops(seg_3d)
    n_removed = 0
    for prop in tqdm.tqdm(props, desc=f"Filtering cells with z-span < {min_z_span}"):
        zmin, _, _, zmax, _, _ = prop.bbox
        if zmax - zmin < min_z_span:
            seg_3d_filtered[seg_3d == prop.label] = 0
            n_removed += 1
    print(f"  Z-span filter: removed {n_removed} cell(s) spanning < {min_z_span} slices")
    return seg_3d_filtered


def make_color_mask(seg_3d):
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


def stacked_volume_stem(stack_path: str) -> str:
    base = os.path.basename(stack_path)
    if base.endswith("_2D_stacked.tif"):
        return base[: -len("_2D_stacked.tif")]
    return os.path.splitext(base)[0].replace("_2D_stacked", "")


def bridge_gaps(seg_3d, ji_thre=0.3):
    props = regionprops(seg_3d)
    if not props:
        return seg_3d

    cell_zmin = {}
    cell_zmax = {}
    for p in props:
        zmin, _, _, zmax, _, _ = p.bbox
        cell_zmin[p.label] = zmin
        cell_zmax[p.label] = zmax

    zmin_to_labels = {}
    for lbl, zmin in cell_zmin.items():
        zmin_to_labels.setdefault(zmin, []).append(lbl)

    merge_map = {}
    fill_slices = []

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
        best_ji, best_b = 0, None
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
    print(f"  Gap bridging: merged {len(merge_map)} fragment(s), filled {len(fill_slices)} missing slice(s)")
    return seg_3d


def split_disconnected_3d(seg_3d):
    props = regionprops(seg_3d)
    next_label = int(seg_3d.max()) + 1
    n_split = 0
    for p in tqdm.tqdm(props, desc="Checking 3D connectivity"):
        zmin, ymin, xmin, zmax, ymax, xmax = p.bbox
        cell_mask = seg_3d[zmin:zmax, ymin:ymax, xmin:xmax] == p.label
        cc, n_cc = sklabel(cell_mask, return_num=True)
        if n_cc <= 1:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        largest = np.argmax(sizes) + 1
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


def filter_size_inconsistent(seg_3d, max_ratio=3.0):
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
        parts.append(f"removed {n_removed} cell(s) entirely")
    if parts:
        print(f"  Size-consistency: {', '.join(parts)}")
    return seg_3d


def filter_small_volumes(seg_3d, min_volume):
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


def absorb_short_fragments(seg_3d, max_short_span=15, ji_thre=0.3, max_iterations=5):
    total_merged = 0
    for _ in range(max_iterations):
        props = regionprops(seg_3d)
        if not props:
            break
        cell_spans = {}
        for p in props:
            zmin, _, _, zmax, _, _ = p.bbox
            cell_spans[p.label] = (int(zmin), int(zmax))

        merge_map = {}
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
            check_before = range(max(0, zmin - 3), zmin)
            check_after = range(zmax, min(seg_3d.shape[0], zmax + 3))

            for z in check_before:
                ref = first_mask
                others = np.unique(seg_3d[z][ref])
                for o in others:
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

            for z in check_after:
                ref = last_mask
                others = np.unique(seg_3d[z][ref])
                for o in others:
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


def relabel_contiguous(seg_3d):
    labels = np.unique(seg_3d)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return seg_3d
    max_label = int(labels.max())
    lut = np.zeros(max_label + 1, dtype=np.int32)
    for new_lbl, old_lbl in enumerate(labels, start=1):
        lut[int(old_lbl)] = new_lbl
    return lut[seg_3d.astype(np.int32)]


def process_stacked_2d_segmentation(seg_2d_stack_path, JI_thre=JI_THRESHOLD, force=False):
    img_name = f"{stacked_volume_stem(seg_2d_stack_path)}_3D"
    rel = os.path.relpath(seg_2d_stack_path, BASE_DIR)
    output_folder = os.path.join(os.fspath(SEGMENTATION_3D_DIR), os.path.dirname(rel))
    index_mask_path = os.path.join(output_folder, f"{img_name}_indexed.tif")
    color_mask_path = os.path.join(output_folder, f"{img_name}_color.tif")

    if not force and os.path.exists(index_mask_path):
        if os.path.exists(color_mask_path):
            print(f"Indexed + color already exist for {seg_2d_stack_path}, skipping...")
            return
        print(f"Indexed mask exists but color missing; writing RGB preview from {index_mask_path}")
        os.makedirs(output_folder, exist_ok=True)
        seg_3d = imread(index_mask_path)
        color_mask = make_color_mask(seg_3d)
        imwrite(color_mask_path, color_mask, photometric="rgb")
        print(f"  Saved {color_mask_path}")
        return

    print("Processing stacked 2D segmentation")
    print(f"  Input:  {seg_2d_stack_path}")
    os.makedirs(output_folder, exist_ok=True)
    seg_2d_stack = imread(seg_2d_stack_path)
    seg_3d = matching_cells_2D(seg_2d_stack, JI_thre=JI_thre)

    seg_3d = bridge_gaps(seg_3d, ji_thre=JI_thre)
    seg_3d = absorb_short_fragments(seg_3d, max_short_span=15, ji_thre=JI_thre)
    seg_3d = filter_short_z_cells(seg_3d, min_z_span=MIN_CELL_Z_SPAN)
    seg_3d = split_disconnected_3d(seg_3d)
    seg_3d = filter_size_inconsistent(seg_3d, max_ratio=MAX_AREA_CHANGE_RATIO)
    seg_3d = filter_small_volumes(seg_3d, min_volume=MIN_CELL_VOLUME_3D)
    seg_3d = filter_short_z_cells(seg_3d, min_z_span=MIN_CELL_Z_SPAN)
    seg_3d = relabel_contiguous(seg_3d)

    tmp_index = index_mask_path + ".tmp"
    tmp_color = color_mask_path + ".tmp"
    imwrite(tmp_index, seg_3d.astype(np.uint16))
    os.replace(tmp_index, index_mask_path)
    color_mask = make_color_mask(seg_3d)
    imwrite(tmp_color, color_mask, photometric="rgb")
    os.replace(tmp_color, color_mask_path)


def _process_force(path):
    return process_stacked_2d_segmentation(path, force=True)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-generate even if outputs already exist")
    args = ap.parse_args()

    stacked_list = glob.glob(os.path.join(BASE_DIR, "**", "*_2D_stacked.tif"), recursive=True)
    if not stacked_list:
        print(f"No *_2D_stacked.tif under {BASE_DIR}")
        return
    np.random.shuffle(stacked_list)
    target = _process_force if args.force else process_stacked_2d_segmentation
    n_workers = min(32, len(stacked_list), (os.cpu_count() or 8))
    with Pool(processes=max(1, n_workers)) as pool:
        pool.map(target, stacked_list)
    print("Processing completed.")


if __name__ == "__main__":
    main()

