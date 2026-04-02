import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from multiprocessing import Pool
from skimage.io import imread
from skimage.measure import regionprops, label as sklabel
from tifffile import imwrite

from segmentation_3D.match_2D_cells import matching_cells_2D

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from pipeline_config import (
    SEGMENTATION_2D_STACKED_DIR,
    SEGMENTATION_3D_DIR,
    JI_THRESHOLD,
    MIN_CELL_VOLUME_3D,
    MAX_AREA_CHANGE_RATIO,
)

base_dir = os.fspath(SEGMENTATION_2D_STACKED_DIR)


def filter_single_slice_cells(seg_3D):
    """Remove cells that only appear in a single z-slice using regionprops."""
    seg_3D_filtered = seg_3D.copy()
    props = regionprops(seg_3D)

    for p in tqdm.tqdm(props, desc="Filtering single-slice cells using regionprops"):
        zmin, ymin, xmin, zmax, ymax, xmax = p.bbox
        if zmax - zmin == 1:
            seg_3D_filtered[seg_3D == p.label] = 0
    return seg_3D_filtered


def make_color_mask(seg_3D):
    """Assign each cell in seg_3D a random color from tab20 colormap."""
    labels = np.unique(seg_3D)
    labels = labels[labels != 0]

    cmap = plt.get_cmap("tab20")
    rng = np.random.default_rng()
    colors = (np.array([cmap(rng.integers(0, 20))[:3] for _ in labels]) * 255).astype(np.uint8)

    lut = np.zeros((seg_3D.max() + 1, 3), dtype=np.uint8)
    lut[labels] = colors

    return lut[seg_3D]


def stacked_volume_stem(stack_path: str) -> str:
    base = os.path.basename(stack_path)
    if base.endswith("_2D_stacked.tif"):
        return base[: -len("_2D_stacked.tif")]
    return os.path.splitext(base)[0].replace("_2D_stacked", "")


# ---------------------------------------------------------------------------
# 3D post-processing helpers
# ---------------------------------------------------------------------------

def bridge_gaps(seg_3D, ji_thre=0.3):
    """Simple post-hoc gap bridging for exactly-one-slice gaps.

    If a cell exists on slice z but is absent on z+1 while a cell with good
    spatial overlap starts on z+2, merge them and fill z+1 with the union of
    the two neighbouring masks.
    """
    props = regionprops(seg_3D)
    if not props:
        return seg_3D

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

        if z_check >= seg_3D.shape[0]:
            continue

        candidates = zmin_to_labels.get(z_check, [])
        if not candidates:
            continue

        mask_a = seg_3D[last_z_a] == label_a

        best_ji, best_b = 0, None
        for label_b in candidates:
            if label_b in merge_map:
                continue
            mask_b = seg_3D[z_check] == label_b
            inter = int(np.sum(mask_a & mask_b))
            union = int(np.sum(mask_a | mask_b))
            if union > 0:
                ji = inter / union
                if ji > best_ji and ji > ji_thre:
                    best_ji = ji
                    best_b = label_b

        if best_b is not None:
            merge_map[best_b] = label_a
            mask_b = seg_3D[z_check] == best_b
            fill_mask = mask_a | mask_b
            fill_slices.append((z_gap, label_a, fill_mask))

    if not merge_map:
        return seg_3D

    max_label = int(seg_3D.max())
    lut = np.arange(max_label + 1, dtype=np.int32)
    for later, earlier in merge_map.items():
        if later <= max_label:
            lut[later] = earlier
    seg_3D = lut[seg_3D.astype(np.int32)]

    for z, label_val, fill_mask in fill_slices:
        bg = seg_3D[z] == 0
        seg_3D[z][fill_mask & bg] = label_val

    print(f"  Gap bridging: merged {len(merge_map)} fragment(s), "
          f"filled {len(fill_slices)} missing slice(s)")
    return seg_3D


def split_disconnected_3d(seg_3D):
    """Split 3D cells whose voxels form multiple disconnected components."""
    props = regionprops(seg_3D)
    next_label = int(seg_3D.max()) + 1
    n_split = 0

    for p in tqdm.tqdm(props, desc="Checking 3D connectivity"):
        zmin, ymin, xmin, zmax, ymax, xmax = p.bbox
        cell_mask = seg_3D[zmin:zmax, ymin:ymax, xmin:xmax] == p.label
        cc, n_cc = sklabel(cell_mask, return_num=True)
        if n_cc <= 1:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        largest = np.argmax(sizes) + 1
        sub = seg_3D[zmin:zmax, ymin:ymax, xmin:xmax]
        for cc_id in range(1, n_cc + 1):
            if cc_id == largest:
                continue
            sub[cc == cc_id] = next_label
            next_label += 1
            n_split += 1

    if n_split:
        print(f"  Split {n_split} disconnected component(s)")
    return seg_3D


def filter_size_inconsistent(seg_3D, max_ratio=3.0):
    """Remove cells whose 2D cross-section area jumps by more than *max_ratio*
    between consecutive occupied z-slices."""
    props = regionprops(seg_3D)
    bad_labels = set()

    for p in props:
        zmin, _, _, zmax, _, _ = p.bbox
        if zmax - zmin <= 1:
            continue
        areas = []
        for z in range(zmin, zmax):
            a = int(np.sum(seg_3D[z] == p.label))
            if a > 0:
                areas.append(a)
        if len(areas) < 2:
            continue
        for k in range(1, len(areas)):
            lo, hi = sorted((areas[k - 1], areas[k]))
            if lo > 0 and hi / lo > max_ratio:
                bad_labels.add(p.label)
                break

    if bad_labels:
        seg_3D[np.isin(seg_3D, list(bad_labels))] = 0
        print(f"  Size-consistency: removed {len(bad_labels)} cell(s)")
    return seg_3D


def filter_small_volumes(seg_3D, min_volume):
    """Remove cells with fewer than *min_volume* voxels."""
    if min_volume <= 0:
        return seg_3D
    counts = np.bincount(seg_3D.ravel())
    small = np.zeros(len(counts), dtype=bool)
    small[1:] = counts[1:] < min_volume
    if small.any():
        lut = np.arange(len(counts), dtype=np.int32)
        lut[small] = 0
        seg_3D = lut[seg_3D.astype(np.int32)]
        print(f"  Volume filter: removed {int(small.sum())} cell(s)")
    return seg_3D


def relabel_contiguous(seg_3D):
    """Relabel cells to contiguous integers 1..N."""
    labels = np.unique(seg_3D)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return seg_3D
    max_label = int(labels.max())
    lut = np.zeros(max_label + 1, dtype=np.int32)
    for new_lbl, old_lbl in enumerate(labels, start=1):
        lut[int(old_lbl)] = new_lbl
    return lut[seg_3D.astype(np.int32)]


# ---------------------------------------------------------------------------
# Main processing entry point
# ---------------------------------------------------------------------------

def process_stacked_2D_segmentation(seg_2D_stack_path, JI_thre=JI_THRESHOLD):
    img_name = f"{stacked_volume_stem(seg_2D_stack_path)}_3D"
    rel = os.path.relpath(seg_2D_stack_path, base_dir)
    output_folder = os.path.join(os.fspath(SEGMENTATION_3D_DIR), os.path.dirname(rel))
    index_mask_path = os.path.join(output_folder, f"{img_name}_indexed.tif")
    if os.path.exists(index_mask_path):
        print(f"Indexed mask already exists for {seg_2D_stack_path}, skipping...")
        return
    print("Processing stacked 2D segmentation")
    print(f"  Input:  {seg_2D_stack_path}")

    os.makedirs(output_folder, exist_ok=True)

    print("Loading 2D stacked segmentation...")
    seg_2D_stack = imread(seg_2D_stack_path)
    print("Stack shape:", seg_2D_stack.shape)

    seg_3D = matching_cells_2D(seg_2D_stack, JI_thre=JI_thre)
    print(f"  After matching: {len(np.unique(seg_3D))} unique labels")

    print("  Bridging gaps across missing slices...")
    seg_3D = bridge_gaps(seg_3D, ji_thre=JI_thre)

    print("  Filtering single-slice cells...")
    seg_3D = filter_single_slice_cells(seg_3D)

    print("  Splitting disconnected 3D components...")
    seg_3D = split_disconnected_3d(seg_3D)

    print("  Filtering size-inconsistent cells...")
    seg_3D = filter_size_inconsistent(seg_3D, max_ratio=MAX_AREA_CHANGE_RATIO)

    print("  Filtering small-volume cells...")
    seg_3D = filter_small_volumes(seg_3D, min_volume=MIN_CELL_VOLUME_3D)

    print("  Relabeling to contiguous IDs...")
    seg_3D = relabel_contiguous(seg_3D)

    print(f"  Final: {len(np.unique(seg_3D))} unique labels")

    print("  Saving indexed mask...")
    imwrite(os.path.join(output_folder, f"{img_name}_indexed.tif"), seg_3D.astype(np.uint16))

    color_mask = make_color_mask(seg_3D)
    output_color = os.path.join(output_folder, f"{img_name}_color.tif")
    imwrite(output_color, color_mask, photometric="rgb")


def main():
    stacked_list = glob.glob(os.path.join(base_dir, "**", "*_2D_stacked.tif"), recursive=True)
    if not stacked_list:
        print(f"No *_2D_stacked.tif under {base_dir}")
        return
    np.random.shuffle(stacked_list)
    print(f"Found {len(stacked_list)} stack(s), e.g. {stacked_list[:3]}")

    n_workers = min(32, len(stacked_list), (os.cpu_count() or 8))
    with Pool(processes=max(1, n_workers)) as pool:
        pool.map(process_stacked_2D_segmentation, stacked_list)

    print("Processing completed.")


if __name__ == "__main__":
    main()
