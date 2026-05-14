#!/usr/bin/env bash
# Start up to four no-deconv segmentation workers on one GPU node via SSH.
# One sample per GPU (CUDA_VISIBLE_DEVICES=0..3) to avoid the 8-worker oversubscription
# issue on Exclusive_Process GPUs.
#
# "Bigger mem" vs LSF: bsub rusage[mem=...] is an LSF reservation only. On a plain SSH
# session, raise process limits here; host RAM is shared with other users of the node.
#
# Usage (from repo root or anywhere):
#   bash scripts/no_deconv/ssh_nohup_seg_registered_4gpu_one_node.sh
#
# Optional env:
#   CMAP_REPO_ROOT, CMAP_NO_DECONV_INPUT_BASE, CMAP_NO_DECONV_OUTPUT_BASE
#   CMAP_SSH_NODE              (default: nodegpu315)
#   CMAP_SSH_DATASET           (default: 4_18_25)
#   CMAP_SSH_SAMPLES           — four space-separated sample dir names (see defaults below)
#   CMAP_SSH_LOG_DIR           — where nohup *.out / launch.log go (default under repo logs/)

set -euo pipefail

PROJECT_ROOT="${CMAP_REPO_ROOT:-/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

INPUT_BASE="${CMAP_NO_DECONV_INPUT_BASE:-/research/dept/dnb/core_operations/ImageAnalysisScratch/Gutierrez/CMAP_general/No_decon_tests/outputs/rough_registration_batch}"
INPUT_BASE="$(cd "$INPUT_BASE" && pwd)"

OUTPUT_BASE="${CMAP_NO_DECONV_OUTPUT_BASE:-$PROJECT_ROOT/output_no_deconv}"
mkdir -p "$OUTPUT_BASE"
OUTPUT_BASE="$(cd "$OUTPUT_BASE" && pwd)"

INNER="${PROJECT_ROOT}/scripts/no_deconv/_seg_registered_one_sample_inner.sh"
NODE="${CMAP_SSH_NODE:-nodegpu315}"
DATASET="${CMAP_SSH_DATASET:-4_18_25}"
# Defaults: four of the remaining CGN positions (not Sample1 Positions 0,2,3,5,6 on LSF).
SAMPLES="${CMAP_SSH_SAMPLES:-CGNSample1_Position8 CGNSample1_Position9 CGNSample2_Position1 CGNSample2_Position3}"

LOG_DIR="${CMAP_SSH_LOG_DIR:-${PROJECT_ROOT}/logs/no_deconv_seg_registered/ssh_nohup_4gpu_${NODE}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"

read -r -a SAMPLE_ARR <<<"$SAMPLES"
if [[ "${#SAMPLE_ARR[@]}" -lt 1 || "${#SAMPLE_ARR[@]}" -gt 4 ]]; then
  echo "ERROR: provide 1–4 samples in CMAP_SSH_SAMPLES (got ${#SAMPLE_ARR[@]})." >&2
  exit 1
fi

chmod +x "$INNER" 2>/dev/null || true

meta="${LOG_DIR}/launch_meta.txt"
{
  echo "node=$NODE"
  echo "dataset=$DATASET"
  echo "samples=${SAMPLE_ARR[*]}"
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "INPUT_BASE=$INPUT_BASE"
  echo "OUTPUT_BASE=$OUTPUT_BASE"
  echo "log_dir=$LOG_DIR"
} | tee "$meta"

ssh -o BatchMode=yes -o ConnectTimeout=30 "$NODE" \
  bash -s _ \
  "${SAMPLE_ARR[@]}" \
  "$PROJECT_ROOT" \
  "$INPUT_BASE" \
  "$OUTPUT_BASE" \
  "$DATASET" \
  "$INNER" \
  "$LOG_DIR" \
  <<'REMOTE'
set -euo pipefail
# Args: _ <1..N sample names> PROJECT_ROOT INPUT_BASE OUTPUT_BASE DATASET INNER LOG_DIR
shift
arr=("$@")
n=$((${#arr[@]} - 6))
if [[ "$n" -lt 1 || "$n" -gt 4 ]]; then
  log_fallback="${arr[$((${#arr[@]} - 1))]}"
  echo "ERROR: expected 1–4 sample args before six path args (got n=$n, argc=${#arr[@]})" | tee -a "${log_fallback}/launch.log" 2>/dev/null || echo "ERROR: bad sample/path arg count (n=$n argc=${#arr[@]})" >&2
  exit 1
fi
samples=("${arr[@]:0:n}")
PROJECT_ROOT="${arr[n]}"
INPUT_BASE="${arr[n+1]}"
OUTPUT_BASE="${arr[n+2]}"
DATASET="${arr[n+3]}"
INNER="${arr[n+4]}"
LOG_DIR="${arr[n+5]}"
echo "=== remote launch $(hostname) $(date -Is) ===" | tee -a "$LOG_DIR/launch.log"

for i in "${!samples[@]}"; do
  gpu="$i"
  sam="${samples[i]}"
  reg="${INPUT_BASE}/${DATASET}/${sam}/registered"
  merge="${OUTPUT_BASE}/${DATASET}/${sam}"
  for f in "$reg/488nm_registered.tif" "$reg/560nm_registered.tif" "$reg/642nm_reference.tif"; do
    if [[ ! -f "$f" ]]; then
      echo "ERROR: missing $f" | tee -a "$LOG_DIR/launch.log"
      exit 1
    fi
  done
  out="${LOG_DIR}/nohup_${sam}_gpu${gpu}.out"
  echo "Starting gpu=${gpu} sample=${sam} -> $out" | tee -a "$LOG_DIR/launch.log"
  nohup bash -lc "set -euo pipefail
    eval \"\$(conda shell.bash hook)\"
    conda activate cmap
    ulimit -n 1048576 2>/dev/null || true
    ulimit -Ss unlimited 2>/dev/null || true
    ulimit -l unlimited 2>/dev/null || true
    export CUDA_VISIBLE_DEVICES=${gpu}
    export PROJECT_ROOT=\"${PROJECT_ROOT}\"
    export NO_DECONV_REGISTERED=\"${reg}\"
    export NO_DECONV_MERGE=\"${merge}\"
    export NO_DECONV_FORCE_OVERWRITE=\"${NO_DECONV_FORCE_OVERWRITE:-0}\"
    exec bash \"${INNER}\"
  " >"$out" 2>&1 &
  echo "  pid=$!" | tee -a "$LOG_DIR/launch.log"
done

echo "=== background workers started $(date -Is) ===" | tee -a "$LOG_DIR/launch.log"
REMOTE

echo "Local log dir: $LOG_DIR"
echo "Tail: ssh $NODE 'tail -f $LOG_DIR/nohup_*_gpu*.out'"
