import os
import re
import glob
import tifffile as tiff
import numpy as np
import pandas as pd
from skimage.measure import regionprops, label

# New imports for parallel and progress
import multiprocessing as mp
from tqdm import tqdm

# User paths and channel map

IMAGE_DIR = "/media/core/core_operations/ImageAnalysisScratch/Schwarz/Jimin/20X Images"


# SELECTED_CHANNEL = "TH"
# if SELECTED_CHANNEL == "TH":
    # base_dir = "/media/core/core_operations/ImageAnalysis/Core/Haoran/core_projects/multi-scale-cellpose/output_Jimin_20X/"
# if SELECTED_CHANNEL == "AT8":
    # base_dir = "/media/core/core_operations/ImageAnalysis/Core/Haoran/core_projects/multi-scale-cellpose/output_Jimin_20X_new/"
base_dir = "/media/core/core_operations/ImageAnalysis/Core/Haoran/core_projects/multi-scale-cellpose/output_Jimin_20X/"
CHANNEL_NAMES = {
    0: "DAPI",
    1: "TH",
    2: "AT8"
}
segmentation_glob = os.path.join(base_dir, "segmentation_3D_masks", "**", "*indexed.tif")
output_dir = os.path.join(base_dir, "cell_intensity_stats")
os.makedirs(output_dir, exist_ok=True)

def load_tiff(path):
    arr = tiff.imread(path)
    return np.asarray(arr)

def guess_common_id(mask_path):
    """
    Extract a reasonable sample identifier from the mask filename.
    Example: 'Mouse01_section3_indexed.tif' -> 'Mouse01_section3'
    """
    name = os.path.basename(mask_path)
    name = re.sub(r'_indexed\.tif$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\.tif{1,2}$', '', name, flags=re.IGNORECASE)
    return name

def find_image_candidates(mask_path):
    """
    Given a mask path like:
      /.../segmentation_3D_masks/PS19 Mice/4 Months/184/184 Slide 1 Section A2/184 Slide 1 Section A2_3D.tif_indexed.tif
    find the corresponding image:
      /.../20X Images/PS19 Mice/4 Months/184/184 Slide 1 Section A2.tif
    """
    # Define the base roots
    # if SELECTED_CHANNEL == "TH":
        # mask_root = "/media/core/core_operations/ImageAnalysis/Core/Haoran/core_projects/multi-scale-cellpose/output_Jimin_20X/segmentation_3D_masks"
    # elif SELECTED_CHANNEL == "AT8":
        # mask_root = "/media/core/core_operations/ImageAnalysis/Core/Haoran/core_projects/multi-scale-cellpose/output_Jimin_20X_new/segmentation_3D_masks"
    mask_root = "/media/core/core_operations/ImageAnalysis/Core/Haoran/core_projects/multi-scale-cellpose/output_Jimin_20X/segmentation_3D_masks"

    image_root = "/media/core/core_operations/ImageAnalysisScratch/Schwarz/Jimin/20X Images"

    # Compute relative path from mask root
    rel_path = os.path.relpath(mask_path, mask_root)
    # Drop the mask filename, keep the relative folder
    rel_dir = os.path.dirname(rel_path)

    # Extract the image filename stem
    mask_filename = os.path.basename(mask_path)
    # Remove the "_3D.tif_indexed.tif" suffix pattern
    image_stem = re.sub(r'_3D\.tif_indexed\.tif$', '', mask_filename)

    # Build the expected image path
    image_path = os.path.join(image_root, rel_dir + ".tif")

    if os.path.exists(image_path):
        return {"mode": "multichannel", "paths": image_path}
    else:
        print(f"Warning: Image not found for mask: {mask_path}")
        print(f"Tried path: {image_path}")
        return {"mode": None, "paths": None}

def split_channels(img):
    """
    Accepts arrays in any of these shapes:
      Z,Y,X
      C,Z,Y,X
      Z,Y,X,C
    Returns dict {channel_index: 3D array} and inferred channel count
    """
    arr = np.asarray(img)
    if arr.ndim == 3:
        # Single channel
        return {0: arr}, 1
    if arr.ndim == 4:
        # Try Z, C, Y, X
        if arr.shape[1] <= 8 and arr.shape[1] < arr.shape[2]:
            chs = {i: arr[:, i] for i in range(arr.shape[1])}
            return chs, arr.shape[1]
    raise ValueError(f"Unsupported image shape {arr.shape}. Expected Z,Y,X or C,Z,Y,X or Z,Y,X,C.")

def compute_props_for_mask(mask_zyx, channels_dict):
    """
    mask_zyx is 3D labeled mask
    channels_dict maps channel_name -> 3D intensity image aligned to mask
    Returns DataFrame with columns:
      cell_id, centroid_x, centroid_y, volume_voxels, mean_<channel>
    """
    mask = np.asarray(mask_zyx)
    if mask.ndim != 3:
        raise ValueError("Mask must be 3D")

    # Basic geometry props from an intensity-free pass
    props_geom = regionprops(mask)
    if not props_geom:
        return pd.DataFrame(columns=["cell_id", "centroid_x", "centroid_y", "volume_voxels"])

    # Gather geometry
    cell_ids = [p.label for p in props_geom]
    centroids = [p.centroid for p in props_geom]  # (z, y, x)
    volumes = [p.area for p in props_geom]        # voxel count

    df = pd.DataFrame({
        "cell_id": cell_ids,
        "centroid_x": [float(c[2]) for c in centroids],
        "centroid_y": [float(c[1]) for c in centroids],
        "volume_voxels": volumes
    })

    # For each channel compute mean intensity per cell via regionprops with intensity_image
    for ch_name, ch_img in channels_dict.items():
        # Validate shape alignment
        if ch_img.shape != mask.shape:
            raise ValueError(f"Channel {ch_name} shape {ch_img.shape} does not match mask {mask.shape}")
        ch_props = regionprops(mask, intensity_image=ch_img)
        mean_by_label = {p.label: float(p.mean_intensity) for p in ch_props}
        df[f"mean_{ch_name}"] = [mean_by_label.get(cid, np.nan) for cid in df["cell_id"]]

    return df

# Worker that processes a single mask path
def process_one(mask_path):
    try:
        common_id = guess_common_id(mask_path)
        found = find_image_candidates(mask_path)

        if found["mode"] is None:
            msg = f"Warning: No images found for {common_id}. Skipping."
            return (msg, False, 0)

        # Load mask
        mask = load_tiff(mask_path)
        # If the mask is not labeled but binary, label it
        if mask.dtype == bool or np.array_equal(np.unique(mask), [0, 1]):
            mask = label(mask)

        # Load channels aligned to mask
        # channels_dict = {}
        # if found["mode"] == "per_channel":
        #     if SELECTED_CHANNEL in found["paths"]:
        #         img = load_tiff(found["paths"][SELECTED_CHANNEL])
        #         channels_dict[SELECTED_CHANNEL] = np.asarray(img, dtype=np.float32)
        #     else:
        #         msg = f"Warning: selected channel {SELECTED_CHANNEL} not found for {common_id}. Skipping."
        #         return (msg, False, 0)

        # elif found["mode"] == "multichannel":
        #     img = load_tiff(found["paths"])
        #     chs, nC = split_channels(img)
        #     picked = None
        #     for idx, img3d in chs.items():
        #         ch_name = CHANNEL_NAMES.get(idx, f"CH{idx}")
        #         if ch_name == SELECTED_CHANNEL:
        #             picked = np.asarray(img3d, dtype=np.float32)
        #             break
        #     if picked is None:
        #         msg = f"Warning: selected channel {SELECTED_CHANNEL} not present in image for {common_id}. Skipping."
        #         return (msg, False, 0)
        #     channels_dict[SELECTED_CHANNEL] = picked

        channels_dict = {}
        if found["mode"] == "per_channel":
            for ch_name in CHANNEL_NAMES.values():
                if ch_name in found["paths"]:
                    img = load_tiff(found["paths"][ch_name])
                    channels_dict[ch_name] = np.asarray(img, dtype=np.float32)
                if not channels_dict:
                    print(f"Warning: Per-channel mode but no channels loaded for {common_id}. Skipping.")
                    continue
        elif found["mode"] == "multichannel":
            img = load_tiff(found["paths"])
            chs, nC = split_channels(img)
            # Map channel indices to desired names if present in CHANNEL_NAMES, else fall back
            for idx, img3d in chs.items():
                ch_name = CHANNEL_NAMES.get(idx, f"CH{idx}")
                channels_dict[ch_name] = np.asarray(img3d, dtype=np.float32)

        df = compute_props_for_mask(mask, channels_dict)

        # Save CSV alongside output_dir with a sensible name
        rel_path = os.path.relpath(mask_path, base_dir)
        rel_dir = os.path.dirname(rel_path)
        output_subdir = os.path.join(output_dir, rel_dir)
        output_subdir = os.path.dirname(output_subdir)
        output_subdir = output_subdir.replace("segmentation_3D_masks/", "")
        os.makedirs(output_subdir, exist_ok=True)
        out_name = f"{common_id}_cell_intensity_stats.csv"
        out_path = os.path.join(output_subdir, out_name)
        df.sort_values("cell_id").to_csv(out_path, index=False)
        msg = f"Wrote {out_path} with {len(df)} rows."
        return (msg, True, len(df))

    except Exception as e:
        msg = f"Error processing {mask_path}: {e}"
        return (msg, False, 0)

def main():
    # Keep your existing slice
    mask_paths = glob.glob(segmentation_glob, recursive=True)
    if not mask_paths:
        print("No mask files found for pattern:", segmentation_glob)
        return

    n_workers = 32
    # Optional chunksize choice to reduce overhead on many files
    chunksize = max(1, len(mask_paths) // (n_workers * 4) or 1)

    with mp.Pool(processes=n_workers) as pool:
        for msg, ok, nrows in tqdm(
            pool.imap_unordered(process_one, mask_paths, chunksize=chunksize),
            total=len(mask_paths),
            desc="Processing masks",
        ):
            print(msg)

if __name__ == "__main__":
    # Safe for all platforms
    # mp.set_start_method("spawn", force=True)  # uncomment if you prefer spawn
    main()
