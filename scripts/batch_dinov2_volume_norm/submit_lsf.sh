#!/usr/bin/env bash
# Submit one LSF job per sample line (CPU job: read combined TIFF, write CSV).
#
# Usage:
#   export CMAP_ROOT=/research_jude/.../cmap   # absolute
#   bash "$CMAP_ROOT/scripts/batch_dinov2_volume_norm/discover_samples.sh" > "$CMAP_ROOT/scripts/batch_dinov2_volume_norm/samples.txt"
#   bash "$CMAP_ROOT/scripts/batch_dinov2_volume_norm/submit_lsf.sh" "$CMAP_ROOT/scripts/batch_dinov2_volume_norm/samples.txt"
#
# Override queue / memory / walltime (single string, no nested double-quotes):
#   export LSF_RESOURCES='-q standard -W 2:00 -n 1 -R rusage[mem=65536]'
#
# Recompute even when CSV exists (for every job):
#   export FORCE_VOLUME_NORM=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export CMAP_ROOT

LIST="${1:-$SCRIPT_DIR/samples.txt}"
if [[ ! -f "$LIST" ]]; then
  echo "Usage: $0 /path/to/samples.txt" >&2
  echo "  Each line: sample path relative to output/, e.g. 4_18_25/CGNSample1_Position0_decon_dsr" >&2
  exit 1
fi

LOGDIR="${LSF_LOG_DIR:-$CMAP_ROOT/scripts/batch_dinov2_volume_norm/logs}"
mkdir -p "$LOGDIR"

# ~5.4 GiB float32 volume + decompression + NumPy → request generous RAM.
# (No inner double-quotes: they break when the whole string is re-expanded.)
LSF_RESOURCES_DEFAULT="-q standard -W 2:00 -n 1 -R rusage[mem=65536]"

while IFS= read -r SAMPLE || [[ -n "$SAMPLE" ]]; do
  [[ -z "${SAMPLE// }" ]] && continue
  [[ "$SAMPLE" =~ ^# ]] && continue
  SAMPLE="${SAMPLE//$'\r'/}"
  SAMPLE="${SAMPLE//[[:space:]]/}"

  SAFE="${SAMPLE//\//_}"
  SAFE="${SAFE//[^A-Za-z0-9._-]/_}"
  if [[ ${#SAFE} -gt 100 ]]; then
    SAFE="${SAFE:0:100}"
  fi

  echo "bsub: $SAMPLE"

  # shellcheck disable=SC2086
  bsub \
    -J "vnorm_${SAFE}" \
    -oo "${LOGDIR}/vnorm_${SAFE}.out" \
    -eo "${LOGDIR}/vnorm_${SAFE}.err" \
    ${LSF_RESOURCES:-$LSF_RESOURCES_DEFAULT} \
    bash "$SCRIPT_DIR/run_one_sample.sh" "$SAMPLE"

done < "$LIST"

echo "Submitted jobs for samples in $LIST"
echo "Logs: $LOGDIR"
