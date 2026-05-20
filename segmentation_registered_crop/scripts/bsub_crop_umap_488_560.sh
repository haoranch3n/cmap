#!/usr/bin/env bash
# UMAP grid search + publish-best for the registered-crop 488+560 union pipeline.
#
# Prerequisites:
#   - bsub_prepare_extract_crop_488_560.sh completed
#     (all cell_qc_union_488_560/qc_features.csv exist)
#
# Output:
#   <OUTPUT_ROOT>/cell_qc_all/canonical_embedding_union_488_560_registered_all.csv
#
# Usage:
#   bsub < segmentation_registered_crop/scripts/bsub_crop_umap_488_560.sh
#   # or run directly (no bsub):
#   bash segmentation_registered_crop/scripts/bsub_crop_umap_488_560.sh
#
# Env overrides:
#   CMAP_SEG_CROP_OUTPUT_ROOT   default: <repo>/output_registered_crop_seg
#   CMAP_SKIP_PUBLISH           set to 1 to omit --publish-best

#BSUB -J crop_umap_488_560
#BSUB -q standard
#BSUB -n 4
#BSUB -W 60
#BSUB -M 16000
#BSUB -R "rusage[mem=16000] span[hosts=1]"
#BSUB -o /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/logs/crop_umap_488_560_%J.out
#BSUB -e /research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap/logs/crop_umap_488_560_%J.err

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
OUTPUT_ROOT="${CMAP_SEG_CROP_OUTPUT_ROOT:-${PROJECT_ROOT}/output_registered_crop_seg}"
SKIP_PUBLISH="${CMAP_SKIP_PUBLISH:-0}"
UMAP_PY="${PROJECT_ROOT}/visualization/canonical_umap_grid_search.py"

echo "=========================================="
echo "UMAP sweep: registered crop (union_488_560_registered)"
echo "OUTPUT_ROOT:  ${OUTPUT_ROOT}"
echo "SKIP_PUBLISH: ${SKIP_PUBLISH}"
echo "Started:      $(date)"
echo "Host:         $(hostname)"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate cmap
cd "${PROJECT_ROOT}"

echo
echo ">>> canonical_umap_grid_search --run-all-local  variant=union_488_560_registered"
python3 -u "${UMAP_PY}" \
  --variant union_488_560_registered \
  --output-root "${OUTPUT_ROOT}" \
  --run-all-local

if [[ "${SKIP_PUBLISH}" != "1" ]]; then
  echo
  echo ">>> canonical_umap_grid_search --publish-best  variant=union_488_560_registered"
  python3 -u "${UMAP_PY}" \
    --variant union_488_560_registered \
    --output-root "${OUTPUT_ROOT}" \
    --publish-best
else
  echo "  SKIP_PUBLISH=1 — skipping --publish-best"
fi

echo
echo "CSV: ${OUTPUT_ROOT}/cell_qc_all/canonical_embedding_union_488_560_registered_all.csv"
echo

# ── RGB artifacts ──────────────────────────────────────────────────────────
MASTER_CSV="${OUTPUT_ROOT}/cell_qc_all/canonical_embedding_union_488_560_registered_all.csv"
if [[ -f "${MASTER_CSV}" ]]; then
  echo ">>> compute_umap_rgb_artifacts.py"
  CMAP_QC_ALL_CSV="${MASTER_CSV}" \
  PYTHONPATH="${PROJECT_ROOT}/napari-plugin/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -u "${PROJECT_ROOT}/scripts/compute_umap_rgb_artifacts.py" \
    --output-root "${OUTPUT_ROOT}"
  echo "  RGB artifacts written to ${OUTPUT_ROOT}/cell_qc_all/"
else
  echo "  SKIP RGB: master CSV not found (UMAP publish step may have failed)."
fi

echo
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
