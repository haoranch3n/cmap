#!/usr/bin/env bash
# Submit one LSF job per sample: crop + extract for strict-overlap union after
# bg_sigma:3 (488+560) pass mask (see scripts/_canonical_union_strict_crop_extract_one.sh).
#
# Prereq: qc/submit_bg_sigma_488560_batch.sh has produced per sample:
#   union_488_560_strict_overlap_pass_bg_sigma_488560_shape.tif
#
# Usage:
#   bash scripts/submit_bsub_canonical_strict_crop_extract_all.sh [--dry-run]
#
# Optional:
#   CMAP_REPO_ROOT   — cmap repo (absolute)
#   CMAP_SUBMIT_LIMIT — max jobs to submit (debug)

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
PROJECT_ROOT="${CMAP_REPO_ROOT:-$PROJECT_ROOT}"
WORKER="${PROJECT_ROOT}/scripts/_canonical_union_strict_crop_extract_one.sh"
PASS_MASK="union_488_560_strict_overlap_pass_bg_sigma_488560_shape.tif"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

LOG_BASE="${PROJECT_ROOT}/logs/canonical_strict_crop_extract"
mkdir -p "${LOG_BASE}"

submit_count=0
skip_count=0

submit_one() {
  local sample_dir="$1"
  local sample_name
  sample_name=$(basename "${sample_dir}")

  if [[ ! -f "${sample_dir}/${PASS_MASK}" ]]; then
    echo "  SKIP (no ${PASS_MASK}): ${sample_name}"
    (( skip_count++ )) || true
    return
  fi

  if [[ -n "${CMAP_SUBMIT_LIMIT:-}" ]] && [[ "${submit_count}" -ge "${CMAP_SUBMIT_LIMIT}" ]]; then
    echo "Stopping: CMAP_SUBMIT_LIMIT=${CMAP_SUBMIT_LIMIT}"
    return 2
  fi

  local log_dir="${LOG_BASE}/$(basename "$(dirname "${sample_dir}")")"
  mkdir -p "${log_dir}"

  if (( DRY_RUN )); then
    echo "  [dry-run] bsub -J cancrop_${sample_name} ... ${sample_dir}"
  else
    bsub \
      -n 1 \
      -q standard \
      -J "cancrop_${sample_name}" \
      -W 180 \
      -M 48000 \
      -R "rusage[mem=48000] span[hosts=1]" \
      -o "${log_dir}/${sample_name}_%J.out" \
      -e "${log_dir}/${sample_name}_%J.err" \
      bash "${WORKER}" "${sample_dir}"
    echo "  submitted: ${sample_name}"
  fi
  (( submit_count++ )) || true
}

echo "=== Canonical strict crop+extract batch ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Dry run:      ${DRY_RUN}"
echo

for ds in 4_24_25_CGN_6_10_2 4_18_25; do
  echo "--- ${ds} ---"
  for d in "${PROJECT_ROOT}/output/${ds}/"*; do
    [[ -d "$d" ]] || continue
    [[ "$(basename "$d")" == _* ]] && continue
    submit_one "$d" || break
  done
  echo
done

echo "=== Done: submitted=${submit_count}  skipped=${skip_count} ==="
echo "Per-dataset logs: ${LOG_BASE}/<dataset>/"
