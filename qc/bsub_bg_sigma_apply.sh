#!/bin/bash
#BSUB -n 1
#BSUB -q standard
#BSUB -J bg_sigma_apply
#BSUB -W 180
#BSUB -M 32000
#BSUB -R "rusage[mem=32000]"

# Run filter_by_intensity.py with --method bg_sigma:3 on Sample7_Position7
# (4_24_25 CGN), then apply the resulting pass column to write filtered TIFFs.
# This is Phase 3 of the bg-sigma filter experiment (plan: .cursor/plans/filter_dev.md).
# Validation criterion: union_488_560 cell 25 (mean_488 ~ 7e-6) must be filtered out;
# filtered_642 cell 25 (mean_488 ~ 1201) must be retained.

set -uo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
SAMPLE_DIR="${PROJECT_ROOT}/output/4_24_25_CGN_6_10_2/Sample7_Position7_decon_dsr"
METHOD="bg_sigma:3"
PASS_COL="pass_bg_sigma_shape"

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "bg_sigma:3 filter — apply on Sample7_Position7"
echo "Sample:  ${SAMPLE_DIR}"
echo "Method:  ${METHOD}"
echo "Started: $(date)"
echo "Host:    $(hostname)"
echo "=========================================="

run_one() {
  local variant="$1"
  local mask_name="$2"
  local out_name="$3"
  local csv_rel="$4"

  echo
  echo ">>> Variant: ${variant}"
  echo "  filter_by_intensity --method ${METHOD} ..."
  python "${PROJECT_ROOT}/qc/filter_by_intensity.py" \
    --from-volume \
    --method "${METHOD}" \
    --variant "${variant}" \
    --output-dir "${SAMPLE_DIR}" \
    --force 2>&1
  echo "    (filter exit $?)"

  echo "  apply_qc_pass_to_label_mask ..."
  python "${PROJECT_ROOT}/qc/apply_qc_pass_to_label_mask.py" \
    --output-dir "${SAMPLE_DIR}" \
    --variant "${variant}" \
    --pass-column "${PASS_COL}" \
    --mask-name "${mask_name}" \
    --output-name "${out_name}" \
    --csv-rel "${csv_rel}" \
    --overwrite 2>&1
  echo "    (apply exit $?)"

  echo "  Verification (cell 25 status) ..."
  python "${PROJECT_ROOT}/qc/_verify_bg_sigma_apply.py" \
    --sample-dir "${SAMPLE_DIR}" \
    --csv "${SAMPLE_DIR}/${csv_rel}" \
    --output-mask "${SAMPLE_DIR}/${out_name}" \
    --pass-column "${PASS_COL}" \
    --track-cell 25 2>&1
  echo "    (verify exit $?)"
}

run_one "filtered_642" \
        "filtered_642.tif" \
        "filtered_642_pass_bg_sigma_shape.tif" \
        "cell_qc/qc_features_filtered.csv"

run_one "union_488_560" \
        "union_488_560.tif" \
        "union_488_560_pass_bg_sigma_shape.tif" \
        "cell_qc_union_488_560/qc_features_filtered.csv"

echo
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
