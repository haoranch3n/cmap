#!/usr/bin/env bash
# Submit one LSF job per line in samples.txt (same pattern as batching
# extract_features.py with --data-rel per sample).
#
# Usage:
#   cd "$CMAP_ROOT"
#   bash scripts/batch_dinov2/discover_samples.sh > samples.txt   # optional
#   bash scripts/batch_dinov2/submit_lsf.sh samples.txt
#
# Edit the bsub lines below for your site: -q, -gpu, -R rusage, -W, -n.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LIST="${1:-${SCRIPT_DIR}/samples.txt}"

if [[ ! -f "$LIST" ]]; then
  echo "Usage: $0 /path/to/samples.txt" >&2
  echo "  Each line: one sample folder name under PIPELINE_OUTPUT_DIR" >&2
  exit 1
fi

if [[ -f "$SCRIPT_DIR/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/env.sh"
fi

LOGDIR="${LSF_LOG_DIR:-$CMAP_ROOT/scripts/batch_dinov2/logs}"
mkdir -p "$LOGDIR"

while IFS= read -r SAMPLE || [[ -n "$SAMPLE" ]]; do
  [[ -z "${SAMPLE// }" ]] && continue
  [[ "$SAMPLE" =~ ^# ]] && continue
  SAMPLE="${SAMPLE//$'\r'/}"
  SAMPLE="${SAMPLE//[[:space:]]/}"

  echo "bsub: $SAMPLE"

  # --- Site-specific: set LSF_RESOURCES before calling, or edit default below ---
  # Example:
  #   export LSF_RESOURCES='-q gpu -W 8:00 -n 8 -R "rusage[mem=65536]" -gpu "num=1:j_exclusive=yes"'
  LSF_RESOURCES_DEFAULT="-q gpu -W 4:00 -n 4 -gpu num=1:j_exclusive=yes"
  # shellcheck disable=SC2086
  bsub \
    -J "dv2_${SAMPLE}" \
    -oo "${LOGDIR}/dv2_${SAMPLE}.out" \
    -eo "${LOGDIR}/dv2_${SAMPLE}.err" \
    ${LSF_RESOURCES:-$LSF_RESOURCES_DEFAULT} \
    bash "$SCRIPT_DIR/run_one_sample.sh" "$SAMPLE"

done < "$LIST"

echo "Submitted jobs for samples listed in $LIST"
echo "Logs: $LOGDIR"
