#!/usr/bin/env bash
# Union 488/560 Otsu-volume filter chain + DINOv2 for both production datasets:
#   - output/4_18_25
#   - output/4_24_25_CGN_6_10_2
#
# Writes (per sample, union_488_560 variant):
#   union_488_560_otsu_shape.tif
#   cell_box_otsu_shape_filtered/   and   cell_box_full_z_otsu_shape_filtered/
#   cell_qc_union_488_560/*  (merged qc_features_filtered.csv)
#   dinov2_volume_norm_bounds_union_488_560.csv + dinov2_embeddings.* — written by GPU submit (step 3)
#   DINOv2 reads cell_box_otsu_shape_filtered/  → cell_qc_union_488_560/dinov2_embeddings.npy
#
# Steps per dataset (login node; LSF only):
#   1) submit_bsub_union_488_560_4_24_25.sh  — gap-fill union_488_560.tif + combined
#   2) submit_bsub_union_chain_cpu.sh      — volume QC + apply pass + crops + features + merge (no DINOv2 prep)
#   3) submit_bsub_union_dinov2_gpu.sh      — volume-norm CSV (if needed) + GPU DINOv2 embeddings
#
# Env:
#   CMAP_REPO_ROOT              — cmap repo (default: this tree)
#   CMAP_FORCE_OVERWRITE        — default 1: CPU/GPU steps use --force/--overwrite so
#                                 embeddings and QC outputs refresh after the rename to
#                                 otsu_shape dirs (set 0 to keep skip-if-done behaviour).
#   CMAP_SKIP_UNION             — set 1 to skip step 1 (union volumes already present).
#   CMAP_SKIP_CPU               — set 1 to skip step 2.
#   CMAP_SKIP_GPU               — set 1 to skip step 3.
#   CMAP_OUTPUT_BASE_4_18 / CMAP_DATA_BASE_4_18 — overrides for 4_18 paths
#   CMAP_OUTPUT_BASE_4_24 / CMAP_DATA_BASE_4_24 — overrides for 4_24 paths
#
# No LSF email flags per site convention.

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
PROJECT_ROOT="${CMAP_REPO_ROOT:-$PROJECT_ROOT}"

OUT18="${CMAP_OUTPUT_BASE_4_18:-$PROJECT_ROOT/output/4_18_25}"
DATA18="${CMAP_DATA_BASE_4_18:-/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysisScratch/Gutierrez/CMAP_cropped_copies/4_18_25}"

OUT24="${CMAP_OUTPUT_BASE_4_24:-$PROJECT_ROOT/output/4_24_25_CGN_6_10_2}"
DATA24="${CMAP_DATA_BASE_4_24:-/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysisScratch/Gutierrez/CMAP_cropped_copies/4_24_25_CGN_6_10_2}"

export CMAP_REPO_ROOT="$PROJECT_ROOT"
export CMAP_FORCE_OVERWRITE="${CMAP_FORCE_OVERWRITE:-1}"

run_dataset() {
  local label="$1" out_base="$2" data_base="$3"
  echo ""
  echo "#####################################################################"
  echo "# $label"
  echo "#   CMAP_OUTPUT_BASE=$out_base"
  echo "#   CMAP_DATA_BASE=$data_base"
  echo "#####################################################################"
  export CMAP_OUTPUT_BASE="$out_base"
  export CMAP_DATA_BASE="$data_base"
  export CMAP_DATASET
  CMAP_DATASET="$(basename "$out_base")"

  if [[ "${CMAP_SKIP_UNION:-0}" != 1 ]]; then
    echo "=== [$label] Step 1/3: union volumes (488+560) ==="
    bash "$PROJECT_ROOT/scripts/submit_bsub_union_488_560_4_24_25.sh"
  else
    echo "=== [$label] Step 1/3: SKIP (CMAP_SKIP_UNION=1) ==="
  fi

  if [[ "${CMAP_SKIP_CPU:-0}" != 1 ]]; then
    echo "=== [$label] Step 2/3: CPU Otsu-shape chain ==="
    bash "$PROJECT_ROOT/scripts/submit_bsub_union_chain_cpu.sh"
  else
    echo "=== [$label] Step 2/3: SKIP (CMAP_SKIP_CPU=1) ==="
  fi

  if [[ "${CMAP_SKIP_GPU:-0}" != 1 ]]; then
    echo "=== [$label] Step 3/3: GPU DINOv2 (cell_box_otsu_shape_filtered) ==="
    bash "$PROJECT_ROOT/scripts/submit_bsub_union_dinov2_gpu.sh"
  else
    echo "=== [$label] Step 3/3: SKIP (CMAP_SKIP_GPU=1) ==="
  fi
}

run_dataset "4_18_25" "$OUT18" "$DATA18"
run_dataset "4_24_25_CGN_6_10_2" "$OUT24" "$DATA24"

echo ""
echo "Done submitting both datasets. Monitor:"
echo "  bjobs -J 'u488560_*'   bjobs -J 'u488560chain_*'   bjobs -J 'u488560dinov2_*'"
