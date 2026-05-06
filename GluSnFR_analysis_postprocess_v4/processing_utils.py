"""
iGluSnFR Analysis Postprocessing Utilities

Preprocessing and analysis pipeline for synapse detection in calcium imaging data.
Supports multiple ilastik model configurations:
- zyx models (e.g., sponOnly): Trained with 3D spatial axes
- tyx models (e.g., 22_images): Trained with time + 2D spatial axes

The module automatically detects model type and applies appropriate axis transformations.
"""

import matplotlib
matplotlib.use('Agg')

import time
import math
import os
import re

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage
import cucim.skimage
import dask.array as da
from joblib import Parallel, delayed
from tifffile import imread, imsave, imwrite
from skimage.measure import regionprops_table
import matplotlib.pyplot as plt
import pandas as pd

# ilastik imports
import vigra
from ilastik import app
from ilastik.applets.dataSelection.opDataSelection import PreloadedArrayDatasetInfo

# GPU memory pools
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

# Global flag: when True, skip BKG subtraction / normalization / segmentation
# and load previously saved files instead.
readSegmented = True

# Model type constants
MODEL_TYPE_TYX = "tyx"
MODEL_TYPE_ZYX = "zyx"

LAZYFLOW_TOTAL_RAM_MB = "100000"


# =============================================================================
# Utility helpers
# =============================================================================

def ensure_dir(filepath):
    """Ensure the directory for a filepath exists."""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


def save_image(filepath, data):
    """Save image with automatic directory creation."""
    ensure_dir(filepath)
    imsave(filepath, data)


def save_csv(df, filepath):
    """Save CSV with automatic directory creation."""
    ensure_dir(filepath)
    df.to_csv(filepath)


def mergeArray(blocks, im_size, dtype=None):
    """Merge a list of array blocks into a single pre-allocated array."""
    if dtype is None:
        dtype = blocks[0].dtype if len(blocks) > 0 else np.uint16
    preallocated_array = np.zeros(im_size, dtype=dtype)
    start_index = 0
    for block in blocks:
        end_index = start_index + block.shape[0]
        preallocated_array[start_index:end_index, :, :] = block
        start_index = end_index
    return preallocated_array


def string_to_list_of_tuples(s):
    """Convert string representation of coordinate list to list of (row, col) tuples."""
    s_clean = re.sub(r'[\[\]]', '', s)
    pairs = s_clean.strip().split('\n')
    list_of_tuples = []
    for pair in pairs:
        nums = pair.strip().split()
        if len(nums) == 2:
            try:
                list_of_tuples.append((int(nums[0]), int(nums[1])))
            except ValueError:
                continue
    return list_of_tuples


def map_clusters_to_image(df, W=1000, H=1000):
    """Map cluster IDs onto a 2-D image using coordinate strings in *df*."""
    image = np.zeros((H, W), dtype=int)
    for cluster_id in df['Cluster_ID'].unique():
        all_points = set()
        cluster_points = df[df['Cluster_ID'] == cluster_id]['coords'].apply(string_to_list_of_tuples)
        for points in cluster_points:
            all_points.update(points)
        for point in all_points:
            if 0 <= point[0] < H and 0 <= point[1] < W:
                image[point[0], point[1]] = cluster_id
    return image


# =============================================================================
# Model detection
# =============================================================================

def detect_model_type(model_path):
    """Auto-detect ilastik model type from filename ('tyx' or 'zyx')."""
    model_name = os.path.basename(model_path).lower()
    if "spon" in model_name or "spononly" in model_name:
        return MODEL_TYPE_ZYX
    return MODEL_TYPE_TYX


# =============================================================================
# GPU-accelerated image operations
# =============================================================================

def percentile_cupy(chunk, percentile, footprint, zoom):
    """GPU percentile filter + zoom (used as dask map_blocks worker)."""
    chunk_gpu = cp.asarray(chunk)
    chunk_gpu = cupyx.scipy.ndimage.percentile_filter(chunk_gpu, percentile=percentile, footprint=footprint)
    chunk_gpu = cupyx.scipy.ndimage.zoom(chunk_gpu, zoom=zoom, order=1)
    chunk_cpu = cp.asnumpy(chunk_gpu).astype(np.uint16)
    del chunk_gpu
    return chunk_cpu


def normalize_cupy_numba(chunk, factor, out_chunk):
    """GPU normalise *chunk* in-place into *out_chunk* (uint16)."""
    chunk_gpu = cp.asarray(chunk).astype(cp.float32)
    chunk_gpu *= factor
    cp.clip(chunk_gpu, 0, 65535, out=chunk_gpu)
    chunk_gpu = chunk_gpu.astype(cp.uint16)
    out_chunk[:] = cp.asnumpy(chunk_gpu)
    del chunk_gpu


def subtract_cupy(x1, x2, out=None, return_max=True):
    """GPU subtract *x2* from *x1*, optionally returning the max value."""
    x1_gpu = cp.asarray(x1).astype(cp.float32)
    x2_gpu = cp.asarray(x2).astype(cp.float32)
    _ = cp.subtract(x1_gpu, x2_gpu, out=x1_gpu)
    max_val = None
    if return_max:
        max_val = cp.amax(x1_gpu).get()
    if out is None:
        out = np.zeros(x1_gpu.shape, dtype=np.float32)
    out[:] = cp.asnumpy(x1_gpu)
    del x1_gpu, x2_gpu
    return out, max_val


def label_cucim(segBlock, out=None):
    """GPU connected-component labelling via cuCIM."""
    ndims = len(segBlock.shape)
    block_gpu = cp.asarray(segBlock)
    block_gpu = cucim.skimage.measure.label(block_gpu, return_num=False, connectivity=ndims)
    if out is None:
        out = np.zeros(shape=block_gpu.shape, dtype=np.int32)
    out[:] = cp.asnumpy(block_gpu)
    del block_gpu
    return out


def temporal_color_code(norm_img, color_array_size=256, color_map='nipy_spectral'):
    """Generate RGB temporal colour-coded projections from a 3-D stack."""
    intensityF = 3 * [np.linspace(0, 1, num=color_array_size)]
    intensityF.append(np.ones(color_array_size))
    intensityFactor = np.stack(intensityF, axis=1)
    intFactor_d = cp.asarray(intensityFactor)

    d, h, w = norm_img.shape
    cmap = plt.get_cmap(color_map, lut=d)

    z_color_coded = np.zeros((h, w, 3), dtype=np.uint8)
    z_color_coded_indiv = np.zeros((h, w, 3), dtype=np.uint8)

    z_color_coded_d = cp.zeros((h, w, 3), dtype=cp.uint8)
    z_color_coded_indiv_d = cp.zeros((h, w, 3), dtype=cp.uint8)
    z_color_coded_magnitude_d = cp.zeros((h, w), dtype=cp.float32)

    for i in range(d):
        slice_d = cp.asarray(norm_img[i, :, :]).astype(cp.float32)
        slice_d = cp.clip(cp.floor(255 * slice_d), 0, 255).astype(cp.uint32)
        slice_lut_d = cp.stack(cp.array(color_array_size * [cmap(i, bytes=8)]), axis=0) * intFactor_d
        slice_lut_d = cp.floor(slice_lut_d).astype(cp.uint8)

        slice_channel_d_mag = cp.zeros((h, w), dtype=cp.float32)
        for c in range(3):
            slice_channel_d = slice_lut_d[:, c][slice_d]
            slice_channel_d_mag += slice_channel_d.astype(cp.float32) ** 2
        slice_channel_d_mag = cp.sqrt(slice_channel_d_mag)

        for c in range(3):
            slice_channel_d = slice_lut_d[:, c][slice_d]
            z_color_coded_d[:, :, c] = cp.where(
                slice_channel_d_mag > z_color_coded_magnitude_d,
                slice_channel_d, z_color_coded_d[:, :, c])
            z_color_coded_indiv_d[:, :, c] = cp.maximum(slice_channel_d, z_color_coded_indiv_d[:, :, c])

        z_color_coded_magnitude_d = cp.sqrt(
            cp.sum(z_color_coded_d.astype(cp.float32) ** 2, axis=2))
        del slice_d, slice_lut_d, slice_channel_d

    z_color_coded[:] = cp.asnumpy(z_color_coded_d)
    z_color_coded_indiv[:] = cp.asnumpy(z_color_coded_indiv_d)
    del z_color_coded_d, z_color_coded_indiv_d, z_color_coded_magnitude_d
    return z_color_coded, z_color_coded_indiv


# =============================================================================
# ilastik prediction
# =============================================================================

def _ilastik_predict_tyx(img, model_path):
    """Run ilastik pixel classification with tyx axis tags."""
    args = app.parse_args([])
    args.headless = True
    args.readonly = True
    args.project = model_path
    args.export_source = "Simple Segmentation"
    shell = app.main(args)
    data = [{"Raw Data": PreloadedArrayDatasetInfo(
        preloaded_array=np.asarray(img),
        axistags=vigra.defaultAxistags("tyx")
    )}]
    seg = shell.workflow.batchProcessingApplet.run_export(data, export_to_array=True)
    return ((seg[0][:, :, :, 1] > 0.5) * 1).astype(np.uint8)


def _ilastik_predict_zyx(img, model_path):
    """Run ilastik pixel classification with zyx axis tags."""
    args = app.parse_args([])
    args.headless = True
    args.readonly = True
    args.project = model_path
    args.export_source = "Simple Segmentation"
    shell = app.main(args)
    data = [{"Raw Data": PreloadedArrayDatasetInfo(
        preloaded_array=np.asarray(img),
        axistags=vigra.defaultAxistags("zyx")
    )}]
    seg = shell.workflow.batchProcessingApplet.run_export(data, export_to_array=True)
    return ((seg[0][:, :, :, 1] > 0.5) * 1).astype(np.uint8)


def _ilastik_predict(img, dim_labels, model_path):
    """Dispatch to the correct ilastik predict function based on axis labels."""
    if dim_labels == "zyx":
        return _ilastik_predict_zyx(img, model_path)
    return _ilastik_predict_tyx(img, model_path)


def _clean_ilastik_log_dir():
    """Remove stale ilastik session logs before a parallel run.

    ilastik's ``_delete_old_session_logs`` is not concurrency-safe: when
    multiple workers initialise simultaneously, one may ``unlink()`` a
    file that another is also trying to delete, raising
    ``FileNotFoundError``.  Pre-cleaning the directory before launching
    workers eliminates the race.
    """
    log_dir = os.path.join(os.path.expanduser("~"), ".cache", "ilastik", "log")
    if not os.path.isdir(log_dir):
        return
    for fname in os.listdir(log_dir):
        fpath = os.path.join(log_dir, fname)
        try:
            os.remove(fpath)
        except OSError:
            pass


def run_ilastik_predict(image, model_path, frames_block=50, dim_labels=None,
                        ilastik_threads="16", processes=20):
    """Run ilastik segmentation in parallel blocks with auto model-type detection."""
    if dim_labels is None:
        model_type = detect_model_type(model_path)
        dim_labels = model_type
        print(f"Auto-detected model type: {model_type} (from {os.path.basename(model_path)})")

    os.environ["LAZYFLOW_THREADS"] = str(ilastik_threads)
    os.environ["LAZYFLOW_TOTAL_RAM_MB"] = LAZYFLOW_TOTAL_RAM_MB

    first_dim_size = image.shape[0]
    n_blocks = max(1, first_dim_size // frames_block)
    blocks = np.array_split(image, n_blocks, axis=0)
    print(f"Processing {len(blocks)} blocks with {dim_labels} axis configuration...")

    # Pre-clean ilastik log directory to avoid race condition where
    # parallel workers concurrently try to delete the same stale log files
    _clean_ilastik_log_dir()

    results = Parallel(n_jobs=processes)(
        delayed(_ilastik_predict)(block, dim_labels, model_path) for block in blocks
    )
    return np.concatenate(results, axis=0)


# =============================================================================
# Region measurement
# =============================================================================

def measurePerTimePoint(label_img, normalized_img, FrameOffset):
    """Measure region properties per frame (used for initial per-block measurement)."""
    frames = []
    for i in range(label_img.shape[0]):
        tmp = pd.DataFrame(regionprops_table(
            label_img[i, :, :], normalized_img[i, :, :],
            properties=('label', 'centroid', 'mean_intensity', 'area',
                        'axis_major_length', 'axis_minor_length', 'coords')))
        tmp["Slice"] = i + FrameOffset
        frames.append(tmp)
    return pd.concat(frames)


def _process_frame_batch_for_regions(frame_indices, img_3d, region_data, props, label_to_idx):
    """Process a batch of frames for region measurement (parallel worker)."""
    rows = []
    for frame_idx in frame_indices:
        frame = img_3d[frame_idx]
        for rd in region_data:
            label_id = rd['label']
            idx = label_to_idx[label_id]
            mean_intensity = frame[rd['rows'], rd['cols']].mean()
            row = {
                'label': label_id,
                'centroid-0': props['centroid-0'][idx],
                'centroid-1': props['centroid-1'][idx],
                'mean_intensity': mean_intensity,
                'area': props['area'][idx],
                'axis_major_length': props['axis_major_length'][idx],
                'axis_minor_length': props['axis_minor_length'][idx],
                'Slice': frame_idx,
            }
            # Morphology metrics (constant per ROI)
            for morph_key in ('eccentricity', 'solidity', 'perimeter',
                              'circularity'):
                if morph_key in props:
                    row[morph_key] = props[morph_key][idx]
            rows.append(row)
    return rows


def measureRegionsOptimizedParallel(mask_2d, normalized_img_3d, n_jobs=15):
    """Measure region properties once on 2-D mask, then extract intensities in parallel."""
    from skimage.measure import regionprops

    n_frames = normalized_img_3d.shape[0]
    props = regionprops_table(mask_2d, properties=(
        'label', 'centroid', 'area', 'axis_major_length', 'axis_minor_length',
        'eccentricity', 'solidity', 'perimeter'))

    # Derived metric: circularity = 4*pi*area / perimeter^2  (1 = perfect circle)
    perimeters = np.asarray(props['perimeter'], dtype=float)
    areas = np.asarray(props['area'], dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        props['circularity'] = np.clip(np.where(
            perimeters > 0, 4.0 * np.pi * areas / (perimeters ** 2), 0.0), 0.0, 1.0)

    if len(props['label']) == 0:
        return pd.DataFrame()

    print(f"  Computing measurements for {len(props['label'])} regions across {n_frames} frames...")

    regions = regionprops(mask_2d)
    region_data = [{'label': r.label, 'rows': r.coords[:, 0], 'cols': r.coords[:, 1]} for r in regions]
    label_to_idx = {label: i for i, label in enumerate(props['label'])}

    n_batches = min(n_jobs, n_frames)
    frame_batches = np.array_split(np.arange(n_frames), n_batches)

    try:
        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=1)(
            delayed(_process_frame_batch_for_regions)(batch, normalized_img_3d, region_data, props, label_to_idx)
            for batch in frame_batches)
    except Exception as e:
        print(f"  Parallel failed ({e}), falling back to sequential...")
        results = [_process_frame_batch_for_regions(batch, normalized_img_3d, region_data, props, label_to_idx)
                   for batch in frame_batches]

    all_rows = []
    for batch_rows in results:
        all_rows.extend(batch_rows)
    return pd.DataFrame(all_rows)


# =============================================================================
# SNR filtering
# =============================================================================

def filter_clusters_by_trace_snr(df, snr_threshold=3.0, cluster_col='label',
                                  time_col='Slice', signal_col='mean_intensity',
                                  baseline_percentile=25, signal_percentile=90,
                                  snr_start_frame=None, snr_end_frame=None):
    """
    Compute per-cluster SNR and split into accepted / rejected lists.

    SNR = (signal_percentile - baseline_percentile) / baseline_std
    """
    df_labeled = df.copy()
    cluster_snr = {}

    if snr_start_frame is not None or snr_end_frame is not None:
        print(f"SNR calculation using frames {snr_start_frame or 0} to {snr_end_frame or 'end'}")

    for cluster_id in df_labeled[cluster_col].unique():
        trace = df_labeled[df_labeled[cluster_col] == cluster_id][signal_col].values
        if len(trace) == 0:
            cluster_snr[cluster_id] = 0
            continue

        start = snr_start_frame if snr_start_frame is not None else 0
        end = snr_end_frame if snr_end_frame is not None else len(trace)
        end = min(end, len(trace))
        start = min(start, end)
        tw = trace[start:end]

        if len(tw) == 0:
            cluster_snr[cluster_id] = 0
            continue

        baseline = np.percentile(tw, baseline_percentile)
        signal = np.percentile(tw, signal_percentile)
        baseline_threshold = np.percentile(tw, baseline_percentile + 10)
        baseline_values = tw[tw <= baseline_threshold]
        baseline_std = np.std(baseline_values) if len(baseline_values) > 1 else 1.0
        if baseline_std == 0:
            baseline_std = 1.0
        cluster_snr[cluster_id] = (signal - baseline) / baseline_std

    clusters_accepted = [cid for cid, snr in cluster_snr.items()
                         if not np.isnan(snr) and snr >= snr_threshold]
    clusters_rejected = [cid for cid, snr in cluster_snr.items()
                         if np.isnan(snr) or snr < snr_threshold]

    df_labeled['SNR'] = df_labeled[cluster_col].map(cluster_snr)
    df_labeled['SNR_status'] = df_labeled[cluster_col].apply(
        lambda x: 'accepted' if x in clusters_accepted else 'rejected')

    print(f"SNR filtering: {len(clusters_accepted)} accepted, {len(clusters_rejected)} rejected (threshold={snr_threshold})")
    return df_labeled, cluster_snr, clusters_accepted, clusters_rejected


def create_accepted_rejected_masks(mask, clusters_accepted, clusters_rejected):
    """Create separate masks for accepted and rejected clusters."""
    mask_accepted = np.zeros_like(mask)
    mask_rejected = np.zeros_like(mask)
    for cid in clusters_accepted:
        mask_accepted[mask == cid] = cid
    for cid in clusters_rejected:
        mask_rejected[mask == cid] = cid
    return mask_accepted, mask_rejected


def apply_snr_filtering(df, mask, final_centroids, snr_threshold, filepath,
                         input_dir, output_dir, seg_dir, BKGpercentile,
                         snr_start_frame=None, snr_end_frame=None):
    """Apply SNR filtering, save masks / CSVs / visualisations."""

    df_labeled, cluster_snr_values, clusters_accepted, clusters_rejected = filter_clusters_by_trace_snr(
        df, snr_threshold=snr_threshold, cluster_col='label', time_col='Slice',
        signal_col='mean_intensity', baseline_percentile=25, signal_percentile=90,
        snr_start_frame=snr_start_frame, snr_end_frame=snr_end_frame)

    mask_accepted, mask_rejected = create_accepted_rejected_masks(mask, clusters_accepted, clusters_rejected)
    print(f"Accepted mask: {len(np.unique(mask_accepted)) - 1} clusters")
    print(f"Rejected mask: {len(np.unique(mask_rejected)) - 1} clusters")

    # Save labeled CSV (into csv_outputs/per_image_csv subfolder)
    csv_output_dir = os.path.join(output_dir, 'csv_outputs', 'per_image_csv')
    csv_path = filepath.replace('.tif', '_full_SNRlabeled_PERCENTILE.csv').replace(input_dir, csv_output_dir).replace('PERCENTILE', 'percentile' + str(BKGpercentile))
    save_csv(df_labeled, csv_path)

    # Save accepted / rejected masks
    for suffix, m in [("_objects_accepted_PERCENTILE.tif", mask_accepted),
                       ("_objects_rejected_PERCENTILE.tif", mask_rejected)]:
        p = filepath.replace(input_dir, seg_dir).replace(".tif", suffix).replace('PERCENTILE', 'percentile' + str(BKGpercentile))
        save_image(p, np.uint16(m))

    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(30, 10), dpi=100)
    axes[0].imshow(mask)
    axes[0].set_title(f'All Clusters (n={len(clusters_accepted) + len(clusters_rejected)})')
    for i, centroid in enumerate(final_centroids):
        status = 'accepted' if i in clusters_accepted else 'rejected'
        color = 'green' if status == 'accepted' else 'red'
        axes[0].plot(centroid[1], centroid[0], 'x', color=color, markersize=8)
        axes[0].text(centroid[1] + 5, centroid[0], f'{i}', color=color, fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(mask_accepted)
    axes[1].set_title(f'Accepted (n={len(clusters_accepted)}, SNR>={snr_threshold})')
    for cid in clusters_accepted:
        if cid < len(final_centroids):
            c = final_centroids[cid]
            axes[1].plot(c[1], c[0], 'gx', markersize=8)
            axes[1].text(c[1] + 5, c[0], f'{cid}\nSNR:{cluster_snr_values[cid]:.1f}', color='green', fontsize=9)
    axes[1].axis('off')

    axes[2].imshow(mask_rejected)
    axes[2].set_title(f'Rejected (n={len(clusters_rejected)}, SNR<{snr_threshold})')
    for cid in clusters_rejected:
        if cid < len(final_centroids):
            c = final_centroids[cid]
            axes[2].plot(c[1], c[0], 'rx', markersize=8)
            axes[2].text(c[1] + 5, c[0], f'{cid}\nSNR:{cluster_snr_values[cid]:.1f}', color='red', fontsize=9)
    axes[2].axis('off')
    plt.tight_layout()

    vis_path = filepath.replace(input_dir, seg_dir).replace(".tif", "_SNR_filtering_PERCENTILE.png").replace('PERCENTILE', 'percentile' + str(BKGpercentile))
    plt.savefig(vis_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # SNR histogram
    plt.figure(figsize=(10, 6))
    plt.hist(list(cluster_snr_values.values()), bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=snr_threshold, color='r', linestyle='--', label=f'Threshold={snr_threshold}')
    plt.xlabel('SNR'); plt.ylabel('Number of Clusters'); plt.title('SNR Distribution'); plt.legend()
    hist_path = filepath.replace(input_dir, seg_dir).replace(".tif", "_SNR_distribution_PERCENTILE.png").replace('PERCENTILE', 'percentile' + str(BKGpercentile))
    plt.savefig(hist_path, bbox_inches='tight')
    plt.close()

    print("SNR filtering complete.")
    return df_labeled, mask_accepted, mask_rejected, clusters_accepted, clusters_rejected


# =============================================================================
# Main pipeline
# =============================================================================

def ProcessImages(filepath, input_dir, norm_dir, seg_dir, output_dir, model,
                  MaxDistance, BKGpercentile=30, seg_dir_read=None,
                  subtracted_dir=None, min_area=20, max_area=400,
                  snr_threshold=3.0, iou_threshold=0.4,
                  snr_start_frame=None, snr_end_frame=None,
                  n_jobs=15, parallel_backend="threading"):
    """
    Full processing pipeline for a single image file.

    Args:
        filepath: Path to input image
        input_dir: Root input directory (used for path mirroring)
        norm_dir: Directory for normalised images
        seg_dir: Directory for segmentation outputs (write)
        output_dir: Root output directory
        model: Path to ilastik model file
        MaxDistance: Max intra-cluster distance for hierarchical clustering
        BKGpercentile: Percentile for background estimation
        seg_dir_read: Directory to read existing segmentation from (None -> seg_dir)
        subtracted_dir: Directory for BKG-subtracted images (None -> auto)
        min_area: Minimum object area (pixels)
        max_area: Maximum object area (pixels)
        snr_threshold: SNR threshold for accepting clusters
        iou_threshold: IoU threshold for merging overlapping clusters
        snr_start_frame: Start frame for SNR calculation (None = first)
        snr_end_frame: End frame for SNR calculation (None = last)
        n_jobs: Number of parallel workers for region measurement
        parallel_backend: Joblib backend ('threading' or 'loky')
    """
    if seg_dir_read is None:
        seg_dir_read = seg_dir
    if subtracted_dir is None:
        subtracted_dir = os.path.join(os.path.dirname(norm_dir), "BKG_subtracted")
    os.makedirs(subtracted_dir, exist_ok=True)
    D = MaxDistance

    model_type = detect_model_type(model)
    print(f"Processing with model type: {model_type}")
    print(f"Model: {os.path.basename(model)}")
    print(filepath)

    # ------------------------------------------------------------------
    # Step 1-4: Background subtraction + normalisation (or load saved)
    # ------------------------------------------------------------------
    if readSegmented:
        # Load normalised image
        norm_filepath = filepath.replace('.tif', '_BGsubtracted_normalized_NoDownsample_T10_1x1_percentile' + str(BKGpercentile) + '.tif').replace(input_dir, norm_dir)
        if not os.path.exists(norm_filepath):
            norm_filepath = os.path.join(norm_dir, os.path.basename(filepath).replace('.tif', '_BGsubtracted_normalized_NoDownsample_T10_1x1_percentile' + str(BKGpercentile) + '.tif'))
        print(f"Loading normalized image from: {norm_filepath}")
        s_norm = imread(norm_filepath)
        im_size = s_norm.shape
        d, h, w = im_size
        print(f"Loaded normalized image shape: {im_size}")

        # Load subtracted image
        subtracted_filepath = filepath.replace('.tif', '_BGsubtracted_NoDownsample_T10_1x1_percentile' + str(BKGpercentile) + '.tif').replace(input_dir, subtracted_dir)
        subtracted_filepath_canonical = subtracted_filepath
        if not os.path.exists(subtracted_filepath):
            subtracted_filepath = os.path.join(subtracted_dir, os.path.basename(filepath).replace('.tif', '_BGsubtracted_NoDownsample_T10_1x1_percentile' + str(BKGpercentile) + '.tif'))
        if os.path.exists(subtracted_filepath):
            print(f"Loading subtracted image from: {subtracted_filepath}")
            s_subtracted = imread(subtracted_filepath)
        else:
            print(f"Subtracted image not found at {subtracted_filepath}")
            print("Computing background-subtracted image from original...")
            img = imread(filepath)
            chunk_shape = (img.shape[0], 100, 100)
            img_chunked = da.from_array(img, chunks=chunk_shape)
            fp = np.ones((20, 1, 1))
            img_median = img_chunked.map_blocks(percentile_cupy, percentile=BKGpercentile, footprint=fp, zoom=(1, 1, 1), dtype=img_chunked.dtype).compute(num_workers=8)
            mempool.free_all_blocks()
            if img_median.shape[0] > img.shape[0]:
                img_median = img_median[:img.shape[0], :, :]
            if img_median.shape[0] < img.shape[0]:
                img = img[:img_median.shape[0], :, :]
            GPU_mem_GiB = 12
            n_blocks = math.ceil(img.nbytes / (1024**3) / (GPU_mem_GiB / 3.2))
            img_blocks = np.array_split(img, n_blocks, axis=0)
            med_blocks = np.array_split(img_median, n_blocks, axis=0)
            blocks = []
            for bi in range(len(img_blocks)):
                b_out, _ = subtract_cupy(img_blocks[bi], med_blocks[bi])
                blocks.append(b_out)
                mempool.free_all_blocks()
            s_subtracted = mergeArray(blocks, img.shape)
            print(f"Saving subtracted image to: {subtracted_filepath_canonical}")
            save_image(subtracted_filepath_canonical, np.clip(s_subtracted, 0, 65535).astype(np.uint16))
            del img, img_chunked, img_median, blocks
            mempool.free_all_blocks()
    else:
        img = imread(filepath)
        im_size = img.shape
        d, h, w = img.shape
        print(f"Image shape: {im_size}")

        chunk_shape = (d, 100, 100)
        img_chunked = da.from_array(img, chunks=chunk_shape)
        fp = np.ones((20, 1, 1))
        print("Calculate percentile filter by chunks")
        img_median = img_chunked.map_blocks(percentile_cupy, percentile=BKGpercentile, footprint=fp, zoom=(1, 1, 1), dtype=img_chunked.dtype).compute(num_workers=8)
        mempool.free_all_blocks()
        if img_median.shape[0] > img.shape[0]:
            img_median = img_median[:img.shape[0], :, :]
        if img_median.shape[0] < img.shape[0]:
            img = img[:img_median.shape[0], :, :]

        GPU_mem_GiB = 12
        n_blocks = math.ceil(img.nbytes / (1024**3) / (GPU_mem_GiB / 3.2))
        print("Subtract background using GPU")
        t0 = time.time()
        img_blocks = np.array_split(img, n_blocks, axis=0)
        del img
        img_med_blocks = np.array_split(img_median, n_blocks, axis=0)
        del img_median
        max_vals, blocks = [], []
        for i in range(n_blocks):
            b_out, bmax = subtract_cupy(img_blocks[i], img_med_blocks[i])
            blocks.append(b_out)
            max_vals.append(bmax)
        factor = 65535.0 / np.max(max_vals)
        print(f"Subtraction took {time.time() - t0:.1f}s")
        del img_med_blocks, img_blocks
        s_subtracted = mergeArray(blocks, im_size)

        print("Run Normalization")
        processed_blocks = []
        for b in blocks:
            b_out = np.zeros(shape=b.shape, dtype=np.uint16)
            normalize_cupy_numba(b, factor, b_out)
            processed_blocks.append(b_out)
        s_norm = mergeArray(processed_blocks, im_size)

        # Baseline normalisation (use last ~100 frames)
        n_frames = s_norm.shape[0]
        bl_start = max(0, n_frames - 100)
        if bl_start == n_frames:
            bl_start = 0
        snorm_BKG_median = np.median(s_norm[bl_start:n_frames, :, :])
        print(f"snorm_BKG_median={snorm_BKG_median} (frames {bl_start}:{n_frames} of {n_frames})")
        if snorm_BKG_median == 0 or np.isnan(snorm_BKG_median):
            print("WARNING: baseline median is 0 or NaN, using 1.0")
            snorm_BKG_median = 1.0
        s_norm = np.clip(s_norm / snorm_BKG_median * 1024, 0, 65535).astype(np.uint16)

        norm_path = filepath.replace('.tif', '_BGsubtracted_normalized_NoDownsample_T10_1x1_PERCENTILE.tif').replace(input_dir, norm_dir).replace('PERCENTILE', 'percentile' + str(BKGpercentile))
        save_image(norm_path, s_norm)
        sub_path = filepath.replace('.tif', '_BGsubtracted_NoDownsample_T10_1x1_PERCENTILE.tif').replace(input_dir, subtracted_dir).replace('PERCENTILE', 'percentile' + str(BKGpercentile))
        save_image(sub_path, np.clip(s_subtracted, 0, 65535).astype(np.uint16))
        print("Finished normalisation")
        del processed_blocks

    # ------------------------------------------------------------------
    # Step 5-6: Segmentation (or load saved)
    # ------------------------------------------------------------------
    if readSegmented:
        t0 = time.time()
        print("Loading segmented image")
        seg_filepath = filepath.replace(input_dir, seg_dir_read).replace('.tif', 'percentile' + str(BKGpercentile) + '.tif')
        if not os.path.exists(seg_filepath):
            seg_filepath = os.path.join(seg_dir_read, os.path.basename(filepath).replace('.tif', 'percentile' + str(BKGpercentile) + '.tif'))
        print(f"Segmented file path: {seg_filepath}")
        ilastik_pred = imread(seg_filepath)
        print(f"Loaded segmented in {time.time() - t0:.1f}s, shape={ilastik_pred.shape}")
    else:
        t0 = time.time()
        ilastik_pred = run_ilastik_predict(s_norm, model)
        print(f"ilastik segmentation took {time.time() - t0:.1f}s")
        seg_path = filepath.replace(input_dir, seg_dir).replace('.tif', 'percentile' + str(BKGpercentile) + '.tif')
        save_image(seg_path, ilastik_pred)

    # ------------------------------------------------------------------
    # Step 7: Temporal colour coding
    # ------------------------------------------------------------------
    color_coded_mag, color_coded = temporal_color_code(ilastik_pred, color_array_size=256, color_map='nipy_spectral')
    for suffix, img_data in [('_color_coded_mag_PERCENTILE.tif', color_coded_mag),
                              ('_color_coded_PERCENTILE.tif', color_coded)]:
        p = filepath.replace('.tif', suffix).replace(input_dir, seg_dir).replace('PERCENTILE', 'percentile' + str(BKGpercentile))
        ensure_dir(p)
        imwrite(p, img_data, photometric='rgb', imagej=True)

    # ------------------------------------------------------------------
    # Step 8: Connected-component labelling
    # ------------------------------------------------------------------
    n_blocks = 20
    seg_blocks = np.array_split(ilastik_pred, n_blocks, axis=0)
    label_output = np.zeros(shape=ilastik_pred.shape, dtype=np.int32)
    label_blocks = np.array_split(label_output, n_blocks, axis=0)
    t0 = time.time()
    for i in range(len(seg_blocks)):
        label_blocks[i] = label_cucim(seg_blocks[i], out=label_blocks[i])
    print(f"Labelling took {time.time() - t0:.1f}s")
    del ilastik_pred, seg_blocks

    # ------------------------------------------------------------------
    # Step 9-10: Region measurement (per time-point)
    # ------------------------------------------------------------------
    s_norm_blocks = np.array_split(s_norm, n_blocks, axis=0)
    s_subtracted_blocks = np.array_split(s_subtracted, n_blocks, axis=0)
    offsets = np.cumsum([b.shape[0] for b in label_blocks])
    offsets = np.insert(offsets, 0, 0)

    df_list = Parallel(n_jobs=n_jobs, backend=parallel_backend, verbose=True)(
        delayed(measurePerTimePoint)(label_blocks[i], s_norm_blocks[i], offsets[i])
        for i in range(len(label_blocks)))
    df = pd.concat(df_list)

    print("Measuring on background-subtracted image...")
    df_sub_list = Parallel(n_jobs=n_jobs, backend=parallel_backend, verbose=True)(
        delayed(measurePerTimePoint)(label_blocks[i], s_subtracted_blocks[i], offsets[i])
        for i in range(len(label_blocks)))
    df_subtracted = pd.concat(df_sub_list)

    df = df.reset_index(drop=True)
    df_subtracted = df_subtracted.reset_index(drop=True)
    df['mean_intensity_subtracted'] = df_subtracted['mean_intensity']

    csv_output_dir = os.path.join(output_dir, 'csv_outputs', 'per_image_csv')
    csv_path = filepath.replace('.tif', '_PERCENTILE.csv').replace(input_dir, csv_output_dir).replace('PERCENTILE', 'percentile' + str(BKGpercentile))
    save_csv(df, csv_path)
    print("Finished per-timepoint measurement")
    del label_blocks, s_norm_blocks, s_subtracted_blocks

    # ------------------------------------------------------------------
    # Step 11: Area filtering
    # ------------------------------------------------------------------
    df = pd.read_csv(csv_path)
    dff = df[(df['area'] > min_area) & (df['area'] < max_area)]
    X = dff[['centroid-0', 'centroid-1']].values
    print(f"Points after area filter: {X.shape[0]}")

    final_labels = None
    final_centroids = None

    if X.shape[0] > 0:
        # --------------------------------------------------------------
        # Step 12: Hierarchical clustering
        # --------------------------------------------------------------
        from scipy.cluster.hierarchy import linkage, fcluster

        print("Computing hierarchical clustering...")
        Z = linkage(X, method='ward')
        merge_distances = Z[:, 2]

        def evaluate_threshold(threshold):
            labels = fcluster(Z, t=threshold, criterion='distance') - 1
            unique_labels = np.unique(labels)
            centroids = np.array([X[labels == lbl].mean(axis=0) for lbl in unique_labels])
            max_dist = 0
            for i, lbl in enumerate(unique_labels):
                pts = X[labels == lbl]
                if len(pts) > 0:
                    dists = np.sqrt(((pts - centroids[i]) ** 2).sum(axis=1))
                    max_dist = max(max_dist, dists.max())
            return labels, centroids, max_dist, len(unique_labels)

        lo = merge_distances.min() * 0.99
        hi = merge_distances.max() * 1.01
        labels_lo, centroids_lo, md_lo, nc_lo = evaluate_threshold(lo)
        labels_hi, centroids_hi, md_hi, nc_hi = evaluate_threshold(hi)

        if md_lo >= D:
            final_labels, final_centroids = labels_lo, centroids_lo
        elif md_hi < D:
            final_labels, final_centroids = labels_hi, centroids_hi
        else:
            while hi - lo > 0.1:
                mid = (lo + hi) / 2
                lm, cm, mdm, ncm = evaluate_threshold(mid)
                if mdm < D:
                    lo = mid
                    final_labels, final_centroids = lm, cm
                else:
                    hi = mid
            if final_labels is None:
                final_labels, final_centroids, _, _ = evaluate_threshold(lo)

        print(f"Clustering: {len(np.unique(final_labels))} clusters (max_dist < {D})")

        # --------------------------------------------------------------
        # Step 13: IoU merging
        # --------------------------------------------------------------
        def get_cluster_coords(df_in, labels, cid):
            pts = set()
            for coord_list in df_in.loc[labels == cid, 'coords'].apply(string_to_list_of_tuples):
                pts.update(coord_list)
            return pts

        unique_clusters = np.unique(final_labels)
        n_clusters = len(unique_clusters)
        cluster_coords_dict = {cid: get_cluster_coords(dff, final_labels, cid) for cid in unique_clusters}

        parent = {cid: cid for cid in unique_clusters}
        def find(x):
            if parent[x] != x: parent[x] = find(parent[x])
            return parent[x]
        def union_op(x, y):
            px, py = find(x), find(y)
            if px != py: parent[px] = py

        clist = list(unique_clusters)
        for i in range(len(clist)):
            for j in range(i + 1, len(clist)):
                if find(clist[i]) != find(clist[j]):
                    c1, c2 = cluster_coords_dict[clist[i]], cluster_coords_dict[clist[j]]
                    if len(c1) > 0 and len(c2) > 0:
                        iou = len(c1 & c2) / len(c1 | c2)
                        if iou > iou_threshold:
                            union_op(clist[i], clist[j])
                            print(f"Merging clusters {clist[i]} and {clist[j]} with IoU={iou:.3f}")

        root_map, counter = {}, 0
        for cid in unique_clusters:
            r = find(cid)
            if r not in root_map:
                root_map[r] = counter; counter += 1
        merged_labels = np.array([root_map[find(c)] for c in final_labels])
        merged_centroids = np.array([X[merged_labels == nid].mean(axis=0) for nid in range(counter)])
        final_labels, final_centroids = merged_labels, merged_centroids
        print(f"After IoU merging: {n_clusters} -> {counter} clusters")

        # --------------------------------------------------------------
        # Step 14-16: Map mask, measure, save
        # --------------------------------------------------------------
        dff = dff.copy()
        dff["Cluster_ID"] = final_labels
        mask = map_clusters_to_image(dff, w, h)
        dff.drop('coords', inplace=True, axis=1)

        csv_output_dir = os.path.join(output_dir, 'csv_outputs', 'per_image_csv')
        events_csv_path = filepath.replace('.tif', '_evetsOnly_withClusterIds_PERCENTILE.csv').replace(input_dir, csv_output_dir).replace('PERCENTILE', 'percentile' + str(BKGpercentile))
        save_csv(dff, events_csv_path)
        objects_path = filepath.replace(input_dir, seg_dir).replace(".tif", "_objects_PERCENTILE.tif").replace('PERCENTILE', 'percentile' + str(BKGpercentile))
        save_image(objects_path, np.uint16(mask))

        # Cluster ID overlay plot
        plt.figure(figsize=(10, 10), dpi=100)
        plt.imshow(mask)
        for i, centroid in enumerate(final_centroids):
            plt.plot(centroid[1], centroid[0], 'wx')
            plt.text(centroid[1] + 5, centroid[0], f'{i}', color='white', fontsize=12)
        plt.axis('off')
        objectIDs_path = filepath.replace(input_dir, seg_dir).replace(".tif", "_objectIDs_PERCENTILE.png").replace('PERCENTILE', 'percentile' + str(BKGpercentile))
        ensure_dir(objectIDs_path)
        plt.savefig(objectIDs_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Step 15-16: Optimised measurement on final mask
        print("Optimized region measurement on final mask...")
        t0 = time.time()
        df = measureRegionsOptimizedParallel(mask, s_norm, n_jobs=n_jobs)
        print(f"Measurement completed in {time.time() - t0:.2f}s")
        print("Measuring on subtracted image...")
        t0 = time.time()
        df_sub = measureRegionsOptimizedParallel(mask, s_subtracted, n_jobs=n_jobs)
        print(f"Subtracted measurement completed in {time.time() - t0:.2f}s")

        df = df.reset_index(drop=True)
        df_sub = df_sub.reset_index(drop=True)
        df['mean_intensity_subtracted'] = df_sub['mean_intensity']

        full_csv_path = filepath.replace('.tif', '_full_PERCENTILE.csv').replace(input_dir, csv_output_dir).replace('PERCENTILE', 'percentile' + str(BKGpercentile))
        save_csv(df, full_csv_path)

        # --------------------------------------------------------------
        # Step 17: SNR filtering
        # --------------------------------------------------------------
        apply_snr_filtering(df, mask, final_centroids, snr_threshold,
                            filepath, input_dir, output_dir, seg_dir, BKGpercentile,
                            snr_start_frame=snr_start_frame, snr_end_frame=snr_end_frame)
