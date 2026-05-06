# Otsu2 Thresholding Implementation Test

## Summary

Successfully implemented and tested a new `otsu2` thresholding method for the QC filter pipeline. This method applies Otsu thresholding on intensity values clipped to the 0-99.99th percentile range, which may reduce over-filtering of cells and retain more valid cells.

## Implementation Details

### Changes Made

**Modified File: `qc/filter_by_intensity.py`**

1. **Added `otsu2` method recognition** in `_parse_method()`
   - Accepts `--method otsu2` flag

2. **Implemented Otsu2 threshold calculation** in `_compute_threshold()`
   - Clips intensity values to the 99.99th percentile
   - Applies standard Otsu thresholding on clipped range
   - Helps reduce impact of extreme outliers

3. **Extended pixel-level thresholding** in `_compute_pixel_thresholds_from_volume()`
   - Added support for otsu2 method in volume-based pixel threshold computation
   - Maintains same methodology as per-cell thresholding

4. **Added dynamic output column naming**
   - Uses method suffix (`otsu` or `otsu2`) in output column names
   - Log_otsu generates: `pass_otsu_shape`
   - Otsu2 generates: `pass_otsu2_shape`

### New Utility Scripts

**`qc/compare_otsu2_methods_separate.py`**
- Compares Otsu and Otsu2 results from separate CSV files
- Displays:
  - Cell pass/fail statistics for each method
  - Threshold value differences per channel
  - Filtered mask cell counts

**`qc/create_otsu2_shape_filtered_mask.py`**
- Creates `filtered_642_pass_otsu2_shape.tif` mask
- Zeros out labels that fail otsu2_shape filter
- Output filename format: `{mask_name}_pass_otsu2_shape.tif`

## Test Results

### Small Sample Test (_smoke_crop_demo)

**Test Dataset:**
- 1 cell
- 13KB filtered mask
- 93KB combined TIFF

**Results:**

```
CSV Statistics:
  Total cells: 1
  Pass log_otsu: 1 (100.00%)
  Pass otsu2:    1 (100.00%)
  Difference:    +0 (  0.00%)

Threshold Comparison (Channel 642):
  log_otsu: 0.0000
  otsu2:    0.0000

Filtered Mask Comparison:
  Log_otsu :   1 pass →   1 labels in mask
  Otsu2    :   1 pass →   1 labels in mask
```

**Output Files Created:**
- `cell_qc/qc_features_filtered_log_otsu.csv` - Original method results
- `cell_qc/qc_features_filtered_otsu2.csv` - Otsu2 method results
- `filtered_642_pass_otsu2_shape.tif` - Filtered mask using otsu2_shape criteria

## Usage

### Running Otsu2 Filter

```bash
python qc/filter_by_intensity.py \
  --from-volume \
  --method otsu2 \
  --variant filtered_642 \
  --output-dir /path/to/sample \
  --force
```

### Creating Filtered Mask

```bash
python qc/create_otsu2_shape_filtered_mask.py \
  /path/to/sample \
  /path/to/sample/cell_qc/qc_features_filtered_otsu2.csv
```

Output: `/path/to/sample/filtered_642_pass_otsu2_shape.tif`

### Comparing Results

```bash
python qc/compare_otsu2_methods_separate.py /path/to/sample
```

## Method Comparison

| Aspect | Log_Otsu | Otsu2 |
|--------|----------|-------|
| Formula | Otsu on log-transformed positive values | Otsu on 0-99.99th percentile clipped values |
| Outlier Sensitivity | Moderate (via log transform) | Low (via explicit clipping) |
| Use Case | Standard QC filtering | Testing impact of outlier exclusion |
| Column Name | `pass_otsu_shape` | `pass_otsu2_shape` |

## Implementation Approach

The otsu2 method addresses potential over-filtering by:

1. **Identifying the 99.99th percentile** intensity value for each channel
2. **Clipping all values** to that range (replacing higher values with the 99th percentile)
3. **Computing Otsu threshold** on the clipped distribution
4. **Comparing results** with standard log_otsu method

This approach reduces the influence of rare high-intensity pixels (noise, hot pixels, artifacts) that might artificially inflate the Otsu threshold.

## Files Modified/Created

### Modified
- `qc/filter_by_intensity.py` - Core implementation

### Created
- `qc/compare_otsu2_methods_separate.py` - Comparison utility
- `qc/create_otsu2_shape_filtered_mask.py` - Mask generation utility
- `qc/bsub_final_otsu2_test.sh` - LSF test script

## Testing on Real Data

To test on a real larger dataset:

```bash
# Request sufficient memory (8GB recommended for 2.8GB TIFF)
bsub -M 8000 -R "rusage[mem=8000]" < qc/bsub_final_otsu2_test.sh
```

The test will:
1. Run log_otsu filter (saves to `qc_features_filtered_log_otsu.csv`)
2. Run otsu2 filter (saves to `qc_features_filtered_otsu2.csv`)
3. Create `filtered_642_pass_otsu2_shape.tif` mask
4. Display detailed comparison report

## Next Steps

1. **Run on production data** to assess impact on cell counts
2. **Adjust percentile threshold** (currently 99.99%) if needed
3. **Integrate into pipeline** if results are promising
4. **Document parameter** for users who want to experiment

## Notes

- The implementation uses separate CSV files for comparison to avoid data loss
- Both methods use the same shape filter (3D morphology gates)
- Threshold values are exported for transparency and debugging
- The `--from-volume` flag is required for this implementation (uses mask + combined TIFF directly)
