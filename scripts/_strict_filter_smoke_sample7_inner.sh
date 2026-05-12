#!/usr/bin/env bash
# Inner worker for _bsub_strict_filter_smoke_sample7.sh.
# Runs filter_by_intensity (bg_sigma:3, 488+560 gating) + apply_qc_pass on the
# strict-overlap union mask for Sample7_Position7_decon_dsr (4_24_25_CGN_6_10_2),
# writing to a side cell_qc folder and a *_smoke output TIFF so legacy artifacts
# are not overwritten. Prints before/after cell counts at the end.

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate cmap

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
SAMPLE_DIR="$PROJECT_ROOT/output/4_24_25_CGN_6_10_2/Sample7_Position7_decon_dsr"
QC_DIR_NAME="cell_qc_union_488_560_strict_smoke"
OUT_NAME="union_488_560_strict_overlap_smoke_pass_bg_sigma_488560_shape.tif"

cd "$PROJECT_ROOT"

echo "=== filter_by_intensity bg_sigma:3 488+560 strict mask ==="
python3 -u "$PROJECT_ROOT/qc/filter_by_intensity.py" \
  --from-volume \
  --method bg_sigma:3 \
  --filter-channels 488,560 \
  --variant union_488_560 \
  --output-dir "$SAMPLE_DIR" \
  --cell-qc-dir-name "$QC_DIR_NAME" \
  --force

echo
echo "=== apply_qc_pass_to_label_mask pass_bg_sigma_488560_shape ==="
python3 -u "$PROJECT_ROOT/qc/apply_qc_pass_to_label_mask.py" \
  --output-dir "$SAMPLE_DIR" \
  --variant union_488_560 \
  --pass-column pass_bg_sigma_488560_shape \
  --csv-rel "$QC_DIR_NAME/qc_features_filtered.csv" \
  --output-name "$OUT_NAME" \
  --overwrite

echo
echo "=== Before / after cell counts ==="
SAMPLE_DIR="$SAMPLE_DIR" QC_DIR_NAME="$QC_DIR_NAME" OUT_NAME="$OUT_NAME" \
python3 -u - <<'PY'
import csv
import os
from pathlib import Path
import numpy as np
import tifffile

SAMPLE = Path(os.environ["SAMPLE_DIR"])
QC = SAMPLE / os.environ["QC_DIR_NAME"] / "qc_features_filtered.csv"
STRICT_MASK = SAMPLE / "union_488_560_strict_overlap.tif"
LEGACY_MASK = SAMPLE / "union_488_560.tif"
OUT_MASK = SAMPLE / os.environ["OUT_NAME"]


def max_label(p):
    if not p.is_file():
        return None
    return int(np.max(tifffile.memmap(p, mode="r")))


legacy_K = max_label(LEGACY_MASK)
strict_K = max_label(STRICT_MASK)

filtered_unique_n = None
if OUT_MASK.is_file():
    arr = tifffile.memmap(OUT_MASK, mode="r")
    uniq = np.unique(arr)
    filtered_unique_n = int((uniq > 0).sum())

n_csv_rows = 0
n_pass = 0
pass_ids = []
fail_ids = []
if QC.is_file():
    with open(QC, newline="") as fh:
        for row in csv.DictReader(fh):
            n_csv_rows += 1
            v = int(float(row.get("pass_bg_sigma_488560_shape", "0") or "0"))
            cid = int(float(row["cell_id"]))
            (pass_ids if v == 1 else fail_ids).append(cid)
            if v == 1:
                n_pass += 1

print()
print(f"  legacy union mask (union_488_560.tif) K              = {legacy_K}")
print(f"  strict overlap mask (union_488_560_strict_overlap.tif) K = {strict_K}")
print(f"  strict QC CSV rows                                   = {n_csv_rows}")
print(f"  strict + bg_sigma:3 488+560 pass count (csv)         = {n_pass}")
print(f"  strict + bg_sigma:3 488+560 pass count (mask uniq>0) = {filtered_unique_n}")
print(f"  strict cell ids PASS  ({len(pass_ids):2d}): {pass_ids}")
print(f"  strict cell ids FAIL  ({len(fail_ids):2d}): {fail_ids}")
PY
