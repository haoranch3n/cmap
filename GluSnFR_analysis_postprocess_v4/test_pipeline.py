#!/usr/bin/env python
"""
iGluSnFR Analysis Toolbox v3 - Pipeline Test Script

Tests the processing pipeline with a sample dataset.

Usage:
    # Run with iglusnfr_processing environment
    ~/.conda/envs/iglusnfr_processing/bin/python test_pipeline.py
    
    # Or with ilastik environment
    ~/.conda/envs/ilastik/bin/python test_pipeline.py

NOTE: Update INPUT_DIR, OUTPUT_DIR, and MODEL_PATH below for your data.
"""

import matplotlib
matplotlib.use('Agg')  # Headless mode

import os
import sys
import time
import psutil

# Add this script's directory to path (processing_utils.py is local)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from processing_utils import ProcessImages
import processing_utils

# =============================================================================
# CONFIGURATION - Update these paths for your data
# =============================================================================

INPUT_DIR = "/path/to/input/images"
OUTPUT_DIR = "/path/to/output"
MODEL_PATH = "/path/to/model.ilp"

# Processing parameters
MAX_DISTANCE = 6
BKG_PERCENTILE = 10
SKIP_SEGMENTATION = True  # Use existing segmented files

# =============================================================================

def resource_monitor():
    """Print resource usage."""
    process = psutil.Process()
    cpu = process.cpu_percent()
    mem = process.memory_info().rss / (1024**3)
    return f"CPU: {cpu:.1f}% | Mem: {mem:.1f} GB"

def main():
    print("=" * 60)
    print("iGluSnFR Analysis Toolbox v3 - Pipeline Test")
    print("=" * 60)
    
    # Validate paths
    if not os.path.isdir(INPUT_DIR) or INPUT_DIR.startswith("/path/to"):
        print("ERROR: Please update INPUT_DIR in this script to point to your image folder.")
        sys.exit(1)
    if not os.path.isfile(MODEL_PATH) or MODEL_PATH.startswith("/path/to"):
        print("ERROR: Please update MODEL_PATH in this script to point to your ilastik model.")
        sys.exit(1)
    
    # Setup directories
    norm_dir = os.path.join(OUTPUT_DIR, "BKG_subtracted_normalized")
    seg_dir = os.path.join(OUTPUT_DIR, "Segmented")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(norm_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    
    # Set skip segmentation flag
    processing_utils.readSegmented = SKIP_SEGMENTATION
    
    # Find images
    from glob import glob
    images = sorted(glob(os.path.join(INPUT_DIR, "**/*.ome.tif"), recursive=True))
    
    print(f"\nInput: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images found: {len(images)}")
    print(f"Skip segmentation: {SKIP_SEGMENTATION}")
    print(f"Resources: {resource_monitor()}")
    print("=" * 60)
    
    if len(images) == 0:
        print("No .ome.tif images found in input directory.")
        sys.exit(1)
    
    # Process each image
    start_time = time.time()
    
    for i, img_path in enumerate(images):
        img_name = os.path.basename(img_path)
        print(f"\n[{i+1}/{len(images)}] Processing: {img_name}")
        print("-" * 40)
        
        try:
            img_start = time.time()
            
            ProcessImages(
                filepath=img_path,
                input_dir=INPUT_DIR,
                norm_dir=norm_dir,
                seg_dir=seg_dir,
                output_dir=OUTPUT_DIR,
                model=MODEL_PATH,
                MaxDistance=MAX_DISTANCE,
                BKGpercentile=BKG_PERCENTILE
            )
            
            img_time = time.time() - img_start
            print(f"Completed in {img_time:.1f}s | {resource_monitor()}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"PIPELINE TEST COMPLETE")
    print(f"Total time: {total_time:.1f}s")
    print(f"Final resources: {resource_monitor()}")
    print("=" * 60)
    
    # Verify outputs
    print("\nOutput files:")
    for pattern, desc in [
        ("csv_outputs/per_image_csv/**/*_full_SNRlabeled_*.csv", "SNR labeled CSV"),
        ("csv_outputs/per_image_csv/**/*_evetsOnly_*.csv", "Events CSV"),
        ("Segmented/**/*_accepted_*.tif", "Accepted ROIs"),
    ]:
        files = glob(os.path.join(OUTPUT_DIR, pattern), recursive=True)
        # Fallback: check legacy (root) location if nothing in new location
        if not files and "csv_outputs" in pattern:
            legacy_pattern = pattern.replace("csv_outputs/per_image_csv/", "")
            files = glob(os.path.join(OUTPUT_DIR, legacy_pattern), recursive=True)
        print(f"  {desc}: {len(files)} files")

if __name__ == "__main__":
    main()
