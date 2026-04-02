import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from multiprocessing import Pool
from skimage.io import imread
from skimage.measure import regionprops
from tifffile import imwrite

from segmentation_3D.match_2D_cells import matching_cells_2D

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from pipeline_config import SEGMENTATION_2D_STACKED_DIR, SEGMENTATION_3D_DIR

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


def process_stacked_2D_segmentation(seg_2D_stack_path, JI_thre=0.3):
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

    print("Before filtering, unique labels:", len(np.unique(seg_2D_stack)))
    seg_3D = filter_single_slice_cells(seg_3D)
    print("After filtering, unique labels:", len(np.unique(seg_3D)))

    print("Saving indexed mask...")
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
