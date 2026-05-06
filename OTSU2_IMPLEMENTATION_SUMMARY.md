# Otsu2 Thresholding Implementation Summary

## Overview

Successfully implemented a new `otsu2` intensity-based QC filtering method that applies Otsu thresholding to intensity values clipped at the 0-99.99th percentile range. This addresses potential over-filtering from outlier pixels.

## What Was Changed

### 1. Core Implementation (`qc/filter_by_intensity.py`)

**Modified Functions:**

- `_parse_method(method_str)` 
  - Added: Recognition of `"otsu2"` method string
  - Added to help message documentation

- `_compute_threshold(values, method, param)`
  - Added: New `otsu2` branch
  - Logic: `clipped = np.clip(values, 0, np.percentile(values, 99.99))`
  - Then: Apply standard `threshold_otsu(clipped)`

- `_compute_pixel_thresholds_from_volume(combined_zcyx, mask, method="log_otsu")`
  - Added: `method` parameter (was hardcoded before)
  - Added: `otsu2` branch for pixel-level thresholding
  - Uses same clipping logic as per-cell method

- `run_from_volume()` function
  - Modified: Method suffix determination: `method_suffix = "otsu2" if method == "otsu2" else "otsu"`
  - Modified: CSV output column names now use method suffix
  - Result: Outputs `pass_otsu2_shape` instead of `pass_qc`

**Output Column Changes:**

When using `--method otsu2`:
- `pass_otsu2_shape` = combined intensity + shape filter pass/fail (replaces `pass_otsu_shape`)
- All threshold columns named with `threshold_{ch}` (same for both methods, different values)

When using `--method log_otsu` (default):
- `pass_otsu_shape` = combined intensity + shape filter pass/fail

## New Utility Scripts

### 1. `qc/compare_otsu2_methods_separate.py`

Compares Otsu and Otsu2 results from two separate CSV files.

**Usage:**
```bash
python qc/compare_otsu2_methods_separate.py /path/to/sample
```

Requires:
- `cell_qc/qc_features_filtered_log_otsu.csv`
- `cell_qc/qc_features_filtered_otsu2.csv`

**Output:**
- Cell pass/fail statistics comparison
- Threshold values per channel
- Filtered mask cell counts
- Difference analysis

### 2. `qc/create_otsu2_shape_filtered_mask.py`

Creates `filtered_642_pass_otsu2_shape.tif` from pass_otsu2_shape column.

**Usage:**
```bash
python qc/create_otsu2_shape_filtered_mask.py /path/to/sample [csv_path]
```

**Output:**
- `filtered_642_pass_otsu2_shape.tif` - Zeros out cells failing otsu2_shape filter

## Test Results (Small Sample)

**Sample:** `_smoke_crop_demo` (1 cell, 93KB combined TIFF)

**Command Sequence:**
```bash
# 1. Run log_otsu
python qc/filter_by_intensity.py --from-volume --method log_otsu \
  --variant filtered_642 --output-dir /path/to/sample --force

# 2. Run otsu2
python qc/filter_by_intensity.py --from-volume --method otsu2 \
  --variant filtered_642 --output-dir /path/to/sample --force

# 3. Create mask
python qc/create_otsu2_shape_filtered_mask.py /path/to/sample \
  /path/to/sample/cell_qc/qc_features_filtered_otsu2.csv

# 4. Compare
python qc/compare_otsu2_methods_separate.py /path/to/sample
```

**Results:**
```
Total cells: 1
Pass log_otsu: 1 (100.00%)
Pass otsu2:    1 (100.00%)
Difference:    +0 (  0.00%)

Both methods passed the single test cell.
Created: filtered_642_pass_otsu2_shape.tif (13KB)
```

## Method Comparison

| Aspect | Log_Otsu | Otsu2 |
|--------|----------|-------|
| **Input Values** | Log-transformed positive voxels | 0-99.99th percentile clipped values |
| **Outlier Handling** | Logarithm reduces extreme value impact | Hard clip removes top 0.01% |
| **Rationale** | Original method, works well generally | Test alternative for outlier sensitivity |
| **Output Column** | `pass_otsu_shape` | `pass_otsu2_shape` |
| **Threshold Behavior** | Moderate sensitivity to high intensities | Lower sensitivity (outlier removal) |

## Files Created

**Modified:**
- `qc/filter_by_intensity.py` - Core implementation

**New:**
- `qc/compare_otsu2_methods_separate.py` - Comparison utility (233 lines)
- `qc/create_otsu2_shape_filtered_mask.py` - Mask creation utility (124 lines)
- `qc/bsub_otsu2_large.sh` - LSF test script
- `qc/bsub_final_otsu2_test.sh` - LSF comprehensive test

**Documentation:**
- `OTSU2_TEST_RESULTS.md` - Test results and usage guide
- `OTSU2_IMPLEMENTATION_SUMMARY.md` - This file

## How to Use

### Single Sample Test

```bash
cd /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap

# Run both methods and compare
python qc/filter_by_intensity.py --from-volume --method log_otsu \
  --variant filtered_642 --output-dir /sample/dir --force
mv /sample/dir/cell_qc/qc_features_filtered.csv \
   /sample/dir/cell_qc/qc_features_filtered_log_otsu.csv

python qc/filter_by_intensity.py --from-volume --method otsu2 \
  --variant filtered_642 --output-dir /sample/dir --force
mv /sample/dir/cell_qc/qc_features_filtered.csv \
   /sample/dir/cell_qc/qc_features_filtered_otsu2.csv

# Create the filtered mask
python qc/create_otsu2_shape_filtered_mask.py /sample/dir \
  /sample/dir/cell_qc/qc_features_filtered_otsu2.csv

# Compare results
python qc/compare_otsu2_methods_separate.py /sample/dir
```

### LSF Batch Test

```bash
cd /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap

# Small sample (fast, low memory)
bsub < qc/bsub_final_otsu2_test.sh

# Large sample (slower, needs ~8-10GB memory)
bsub < qc/bsub_otsu2_large.sh
```

## Expected Outputs

For each sample:
- `cell_qc/qc_features_filtered_log_otsu.csv` - Log_Otsu results
- `cell_qc/qc_features_filtered_otsu2.csv` - Otsu2 results
- `filtered_642_pass_otsu2_shape.tif` - Filtered mask using Otsu2 criteria

## Integration Notes

- Both methods use the same shape filter (3D morphology gates)
- The 99.99th percentile clip level can be adjusted in `_compute_threshold()` if needed
- CSV files must be manually saved/renamed to prevent overwriting
- The implementation maintains backward compatibility (default is still log_otsu)

## Next Steps

1. **Test on production datasets** to see if Otsu2 is more/less stringent
2. **Adjust percentile threshold** (currently 99.99%) based on observed results
3. **Decide which method to use** going forward
4. **Document choice** in pipeline documentation
5. **Consider integrating** one method into automated pipeline if superior

## Technical Details

### Percentile Clipping

The 99.99th percentile was chosen to:
- Remove extreme outliers (top 0.01% of pixels)
- Preserve the main distribution shape
- Reduce Otsu threshold elevation from hot pixels/artifacts

### Shape Filter Interaction

Both methods use the same shape filter:
- 3D bounding box aspect ratio < 16.0
- Fill ratio > 0.045
- Inertia anisotropy ratio < 20.0
- Erosion connectivity test (for large objects)

The combined pass/fail (`pass_otsu2_shape`) requires BOTH intensity AND shape criteria to pass.

## Testing Requirements

- RAM: 2.5GB for small samples (~100MB TIFF)
- RAM: 8-10GB for large samples (2-3GB TIFF)
- Python packages: numpy, tifffile, scipy, scikit-image
- Queue: standard (CPU-only, no GPU needed)
