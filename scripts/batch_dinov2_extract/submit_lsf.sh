#!/usr/bin/env bash
# Submit one LSF GPU job per sample line (DINOv2 extract).
#
# Usage:
#   export CMAP_ROOT=/abs/path/to/cmap
#   export CELL_BOXING_DIR=cell_boxing_filtered   # optional
#   export CELL_QC_DIR=cell_qc_filtered            # optional
#   bash "$CMAP_ROOT/scripts/batch_dinov2_extract/discover_dinov2_extract_samples.py" \
#     4_18_25 --cell-boxing-dir cell_boxing_filtered > samples.txt
#   grep -v '^#' samples.txt > samples_clean.txt
#   bash "$CMAP_ROOT/scripts/batch_dinov2_extract/submit_lsf.sh" samples_clean.txt
#
# Override resources: set the same args you would pass to bsub after -eo … (must stay
# quoted so -R 'rusage[...] span[...]' is one argument; unquoted ${VAR} breaks on '[').
#   export LSF_BSUB_EXTRA='-q other_gpu -W 6:00 -n 1 -R '\''rusage[mem=49152]'\'' -gpu '\''num=1'\'''

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export CMAP_ROOT

CELL_BOXING_DIR="${CELL_BOXING_DIR:-cell_boxing}"
CELL_QC_DIR="${CELL_QC_DIR:-cell_qc}"
# When 1, pass "force" as 4th arg to run_one_sample.sh so LSF jobs overwrite outputs.
FORCE_DINOV2_SUBMIT="${FORCE_DINOV2_SUBMIT:-0}"

LIST="${1:-}"
if [[ ! -f "$LIST" ]]; then
  echo "Usage: $0 /path/to/samples.txt" >&2
  echo "  Each line: sample path relative to output/, e.g. 4_18_25/CGNSample1_Position0_decon_dsr" >&2
  exit 1
fi

LOGDIR="${LSF_LOG_DIR:-$CMAP_ROOT/scripts/batch_dinov2_extract/logs}"
mkdir -p "$LOGDIR"

while IFS= read -r SAMPLE || [[ -n "$SAMPLE" ]]; do
  [[ -z "${SAMPLE// }" ]] && continue
  [[ "$SAMPLE" =~ ^# ]] && continue
  SAMPLE="${SAMPLE//$'\r'/}"
  SAMPLE="${SAMPLE//[[:space:]]/}"

  SAFE="${SAMPLE//\//_}"
  SAFE="${SAFE//[^A-Za-z0-9._-]/_}"
  if [[ ${#SAFE} -gt 80 ]]; then
    SAFE="${SAFE:0:80}"
  fi

  echo "bsub: $SAMPLE  (boxing=$CELL_BOXING_DIR qc=$CELL_QC_DIR force=${FORCE_DINOV2_SUBMIT})"

  extra_args=()
  if [[ "${FORCE_DINOV2_SUBMIT}" == "1" ]]; then
    extra_args=( force )
  fi

  if [[ -n "${LSF_BSUB_EXTRA:-}" ]]; then
    # shellcheck disable=SC2086
    bsub \
      -J "dinov2_${SAFE}" \
      -oo "${LOGDIR}/dinov2_${SAFE}.out" \
      -eo "${LOGDIR}/dinov2_${SAFE}.err" \
      ${LSF_BSUB_EXTRA} \
      bash "$SCRIPT_DIR/run_one_sample.sh" "$SAMPLE" "$CELL_BOXING_DIR" "$CELL_QC_DIR" "${extra_args[@]}"
  else
    bsub \
      -J "dinov2_${SAFE}" \
      -oo "${LOGDIR}/dinov2_${SAFE}.out" \
      -eo "${LOGDIR}/dinov2_${SAFE}.err" \
      -q rhel88_gpu \
      -W 4:00 \
      -n 1 \
      -R 'rusage[mem=32768] span[hosts=1]' \
      -gpu 'num=1:j_exclusive=yes' \
      bash "$SCRIPT_DIR/run_one_sample.sh" "$SAMPLE" "$CELL_BOXING_DIR" "$CELL_QC_DIR" "${extra_args[@]}"
  fi

done < "$LIST"

echo "Submitted jobs for samples in $LIST"
echo "Logs: $LOGDIR"
