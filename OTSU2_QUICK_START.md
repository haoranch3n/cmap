# Otsu2 - Quick Start Guide

## What is Otsu2?

New intensity-based QC filtering method that clips to the 0-99.99th percentile before computing Otsu thresholds. Output includes `_otsu2_shape.tif` mask and `pass_otsu2_shape` column in CSV.

## Test It Now (Small Sample)

```bash
cd /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap

# 1. Run otsu2 filter
python qc/filter_by_intensity.py --from-volume --method otsu2 \
  --variant filtered_642 \
  --output-dir ./output/_smoke_crop_demo --force

# 2. Create filtered mask
python qc/create_otsu2_shape_filtered_mask.py \
  ./output/_smoke_crop_demo

# 3. Check the output
ls -lh ./output/_smoke_crop_demo/filtered_642_pass_otsu2_shape.tif
```

**Expected Result:** ~13KB mask file created

## Test on Your Data

```bash
# 1. Run on your sample (2-3 min for 2.8GB sample)
python qc/filter_by_intensity.py --from-volume --method otsu2 \
  --variant filtered_642 \
  --output-dir /path/to/your/sample --force

# 2. Create mask with pass_otsu2_shape filter
python qc/create_otsu2_shape_filtered_mask.py \
  /path/to/your/sample \
  /path/to/your/sample/cell_qc/qc_features_filtered.csv

# 3. Compare with log_otsu (requires both CSV files)
# Save CSVs separately first
mv /path/to/your/sample/cell_qc/qc_features_filtered.csv \
   /path/to/your/sample/cell_qc/qc_features_filtered_log_otsu.csv

# Run otsu2 again, then:
python qc/compare_otsu2_methods_separate.py /path/to/your/sample
```

## Output Files

After running otsu2:
- `cell_qc/qc_features_filtered.csv` → contains `pass_otsu2_shape` column
- `filtered_642_pass_otsu2_shape.tif` → filtered label mask
- Compare with `filtered_642_pass_otsu_shape.tif` (log_otsu version)

## Key Differences: Otsu2 vs Log_Otsu

| | Log_Otsu | Otsu2 |
|---|----------|-------|
| **Clipping** | Log transform | 99.99th percentile clip |
| **Outlier Sensitivity** | Moderate | Low |
| **CSV Column** | `pass_otsu_shape` | `pass_otsu2_shape` |
| **Mask Suffix** | `pass_otsu_shape.tif` | `pass_otsu2_shape.tif` |

## Cell Count Comparison

Check how many cells pass each method:

```bash
python qc/compare_otsu2_methods_separate.py /your/sample/path
```

Shows:
- How many cells pass log_otsu vs otsu2
- Threshold values per channel
- Cell count in filtered masks
- Percentage difference

## For Large Samples (8-10GB RAM)

```bash
# Submit to LSF queue
cd /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap
bsub < qc/bsub_otsu2_large.sh
```

This will test on the large 2.8GB sample and save results.

## Need Help?

- **Technical details:** Read `OTSU2_IMPLEMENTATION_SUMMARY.md`
- **Test methodology:** Read `OTSU2_TEST_RESULTS.md`
- **Full guide:** Read `DELIVERY_SUMMARY.md`

## Command Cheat Sheet

```bash
# Run otsu2 on sample
python qc/filter_by_intensity.py --from-volume --method otsu2 \
  --variant filtered_642 --output-dir /sample/path --force

# Create mask (requires CSV from above)
python qc/create_otsu2_shape_filtered_mask.py /sample/path

# Compare methods (requires both log_otsu and otsu2 CSVs)
python qc/compare_otsu2_methods_separate.py /sample/path

# Check mask was created
ls -lh /sample/path/filtered_642_pass_otsu2_shape.tif
```

## Expected Results

On small test sample (_smoke_crop_demo):
- Both methods pass the 1 cell
- Thresholds are the same (no extreme values to affect clipping)
- Mask size: ~13KB

On production data:
- Will see difference if there are extreme outlier pixels
- Otsu2 may pass more cells (less sensitive to hot pixels)
- Check comparison report for details

---

**Status: Ready to Test** ✓

Created: May 6, 2026
