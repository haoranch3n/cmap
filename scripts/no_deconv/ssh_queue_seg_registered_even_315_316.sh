#!/usr/bin/env bash
# Fair queue: run no-deconv registered segmentation on nodegpu315 + nodegpu316
# with at most CMAP_QUEUE_MAX_PER_NODE jobs per node (default 4) and
# CMAP_QUEUE_MAX_TOTAL across the node list (default: #nodes × max_per_node, e.g. 8).
# Each job sets CUDA_VISIBLE_DEVICES to the next free slot 0..(max_per_node-1) on that host.
#
# Each poll cycle rebuilds the pending list from disk (merge masks + inputs),
# subtracts LSF inflight ndseg_* names and SSH run_pipeline lines, then fills
# empty slots on the less-loaded node first so 315/316 stay even.
#
# Usage (login node; long-lived — prefer nohup or an LSF sleep job):
#   flock -n ~/cmap_ssh_even_queue.lock bash scripts/no_deconv/ssh_queue_seg_registered_even_315_316.sh
#   CMAP_QUEUE_MODE=once bash scripts/no_deconv/ssh_queue_seg_registered_even_315_316.sh   # one fill pass
#
# Env:
#   CMAP_REPO_ROOT, CMAP_NO_DECONV_INPUT_BASE, CMAP_NO_DECONV_OUTPUT_BASE
#   CMAP_QUEUE_DATASET          (default: 4_18_25) — used if CMAP_QUEUE_DATASETS unset
#   CMAP_QUEUE_DATASETS         — optional: space-separated list (e.g. two batches → 95 samples)
#   CMAP_QUEUE_NODES            (default: "nodegpu315 nodegpu316")
#   CMAP_QUEUE_MAX_PER_NODE     (default: 4)
#   CMAP_QUEUE_MAX_TOTAL        (default: #nodes × max_per_node, e.g. 8 for two nodes)
#   CMAP_QUEUE_POLL_SEC         (default: 90)  — daemon sleep between passes
#   CMAP_QUEUE_MODE             daemon | once (default: daemon)
#   CMAP_QUEUE_LOG_DIR          optional log root (default: logs/no_deconv_seg_registered/ssh_queue_even_<ts>)

set -euo pipefail

PROJECT_ROOT="${CMAP_REPO_ROOT:-/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
INPUT_BASE="${CMAP_NO_DECONV_INPUT_BASE:-/research/dept/dnb/core_operations/ImageAnalysisScratch/Gutierrez/CMAP_general/No_decon_tests/outputs/rough_registration_batch}"
INPUT_BASE="$(cd "$INPUT_BASE" && pwd)"
OUTPUT_BASE="${CMAP_NO_DECONV_OUTPUT_BASE:-$PROJECT_ROOT/output_no_deconv}"
mkdir -p "$OUTPUT_BASE"
OUTPUT_BASE="$(cd "$OUTPUT_BASE" && pwd)"

DATASET="${CMAP_QUEUE_DATASET:-4_18_25}"
if [[ -n "${CMAP_QUEUE_DATASETS:-}" ]]; then
  read -r -a QUEUE_DATASETS <<<"${CMAP_QUEUE_DATASETS}"
else
  QUEUE_DATASETS=("$DATASET")
fi
# pgrep/grep fragment kept for docs; busy detection uses full OUTPUT_BASE prefix on PIPELINE_OUTPUT_DIR.
read -r -a NODES <<<"${CMAP_QUEUE_NODES:-nodegpu315 nodegpu316}"
MAX_PER_NODE="${CMAP_QUEUE_MAX_PER_NODE:-4}"
if [[ -n "${CMAP_QUEUE_MAX_TOTAL:-}" ]]; then
  MAX_TOTAL="${CMAP_QUEUE_MAX_TOTAL}"
else
  MAX_TOTAL=$(( ${#NODES[@]} * MAX_PER_NODE ))
fi
POLL_SEC="${CMAP_QUEUE_POLL_SEC:-90}"
MODE="${CMAP_QUEUE_MODE:-daemon}"
INNER="${PROJECT_ROOT}/scripts/no_deconv/_seg_registered_one_sample_inner.sh"
chmod +x "$INNER" 2>/dev/null || true

LOG_DIR="${CMAP_QUEUE_LOG_DIR:-${PROJECT_ROOT}/logs/no_deconv_seg_registered/ssh_queue_even_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
COORD_LOG="${LOG_DIR}/coordinator.log"

log() { echo "$(date -Is) $*" | tee -a "$COORD_LOG"; }

discover_incomplete() {
  python3 - "$INPUT_BASE" "$OUTPUT_BASE" "${QUEUE_DATASETS[@]}" <<'PY'
import sys
from pathlib import Path

argv = sys.argv[1:]
input_base = Path(argv[0])
output_base = Path(argv[1])
datasets = argv[2:]
chans = ("488nm_crop", "560nm_crop", "642nm_crop")


def merge_done(merge: Path) -> bool:
    return all(
        (merge / ch / "segmentation_3D_masks" / f"{ch}_3D_indexed.tif").is_file()
        for ch in chans
    )


for dataset in datasets:
    in_ds = input_base / dataset
    if not in_ds.is_dir():
        print(f"missing dataset dir: {in_ds}", file=sys.stderr)
        sys.exit(1)
    for pos in sorted(in_ds.iterdir()):
        if not pos.is_dir() or pos.name.startswith("_"):
            continue
        sam = pos.name
        reg = pos / "registered"
        if not reg.is_dir():
            continue
        for fn in ("488nm_registered.tif", "560nm_registered.tif", "642nm_reference.tif"):
            if not (reg / fn).is_file():
                break
        else:
            merge = output_base / dataset / sam
            if not merge_done(merge):
                print(f"{dataset}\t{sam}")
PY
}

# bsub -J ndseg_4_18_25_CGNSample1_Position2 → sample name after prefix (single-dataset only).
lsf_inflight_samples() {
  if [[ "${#QUEUE_DATASETS[@]}" -ne 1 ]]; then
    return 0
  fi
  local prefix="ndseg_${QUEUE_DATASETS[0]}_"
  bjobs -u "$USER" -noheader -o job_name 2>/dev/null | while read -r j; do
    [[ "$j" == "$prefix"* ]] || continue
    printf '%s\t%s\n' "${QUEUE_DATASETS[0]}" "${j#"${prefix}"}"
  done
}

# Busy detection: pgrep only shows `python3 -u .../run_pipeline.py` without argv paths.
# Resolve PIPELINE_OUTPUT_DIR from /proc/<pid>/environ (Linux) and map to dataset<TAB>sample.
ssh_running_samples() {
  local n
  for n in "${NODES[@]}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$n" env OUTPUT_BASE="$OUTPUT_BASE" bash -s <<'EOSH'
set -euo pipefail
ob="${OUTPUT_BASE:?}"
declare -A roots=()
for pid in $(pgrep -f 'segmentation/run_pipeline.py' 2>/dev/null || true); do
  [[ -r "/proc/${pid}/environ" ]] || continue
  line=$(tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | grep '^PIPELINE_OUTPUT_DIR=' || true)
  [[ -z "${line}" ]] && continue
  v="${line#PIPELINE_OUTPUT_DIR=}"
  [[ "${v}" == "${ob}"/* ]] || continue
  mr=$(dirname "${v}")
  roots["${mr}"]=1
done
for mr in "${!roots[@]}"; do
  sample=$(basename "${mr}")
  dataset=$(basename "$(dirname "${mr}")")
  printf '%s\t%s\n' "${dataset}" "${sample}"
done
EOSH
  done | sort -u
}

count_node() {
  local n="$1" c
  c=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$n" env OUTPUT_BASE="$OUTPUT_BASE" bash -s <<'EOSH'
set -euo pipefail
ob="${OUTPUT_BASE:?}"
declare -A roots=()
for pid in $(pgrep -f 'segmentation/run_pipeline.py' 2>/dev/null || true); do
  [[ -r "/proc/${pid}/environ" ]] || continue
  line=$(tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | grep '^PIPELINE_OUTPUT_DIR=' || true)
  [[ -z "${line}" ]] && continue
  v="${line#PIPELINE_OUTPUT_DIR=}"
  [[ "${v}" == "${ob}"/* ]] || continue
  mr=$(dirname "${v}")
  roots["${mr}"]=1
done
echo "${#roots[@]}"
EOSH
  )
  c=$(echo "$c" | tr -d '[:space:]')
  [[ "$c" =~ ^[0-9]+$ ]] || c=0
  echo "$c"
}

compute_pending_lines() {
  local disc busy
  disc=$(discover_incomplete | sort -u)
  busy=$( { lsf_inflight_samples; ssh_running_samples; } | grep -v '^[[:space:]]*$' | sort -u )
  comm -23 <(echo "$disc") <(echo "$busy")
}

launch_on_node() {
  local node="$1" gpu="$2" dataset="$3" sam="$4"
  local reg merge out worker_sh
  reg="${INPUT_BASE}/${dataset}/${sam}/registered"
  merge="${OUTPUT_BASE}/${dataset}/${sam}"
  for f in "$reg/488nm_registered.tif" "$reg/560nm_registered.tif" "$reg/642nm_reference.tif"; do
    if [[ ! -f "$f" ]]; then
      log "SKIP ${dataset}/${sam} (missing input under $reg)"
      return 1
    fi
  done
  out="${LOG_DIR}/nohup_${node}__${dataset}__${sam}_gpu${gpu}.out"
  worker_sh="${LOG_DIR}/_worker_${node}__${dataset}__${sam}_gpu${gpu}.sh"
  cat >"$worker_sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${gpu}
export PROJECT_ROOT="${PROJECT_ROOT}"
export NO_DECONV_REGISTERED="${reg}"
export NO_DECONV_MERGE="${merge}"
export NO_DECONV_FORCE_OVERWRITE="${NO_DECONV_FORCE_OVERWRITE:-0}"
exec bash "${INNER}"
EOF
  chmod +x "$worker_sh"
  log "START $node CUDA_VISIBLE_DEVICES=$gpu ${dataset}/${sam} -> $out"
  local remote_cmd
  remote_cmd="eval \"\$(conda shell.bash hook)\" && conda activate cmap && ulimit -n 1048576 2>/dev/null || true && ulimit -Ss unlimited 2>/dev/null || true && ulimit -l unlimited 2>/dev/null || true && exec bash $(printf %q "$worker_sh")"
  ssh -o BatchMode=yes -o ConnectTimeout=30 "$node" \
    "nohup bash -lc $(printf %q "$remote_cmd") >>$(printf %q "$out") 2>&1 </dev/null &"
}

# Print: <node> <gpu_index> where gpu_index is 0..max_per_node-1 (next free slot on that node).
pick_node_and_gpu() {
  local n0="${NODES[0]}" n1 c0 c1
  c0=$(count_node "$n0")
  if [[ "${#NODES[@]}" -lt 2 ]]; then
    if [[ "$c0" -lt "$MAX_PER_NODE" ]]; then
      echo "$n0" "$c0"
      return 0
    fi
    return 1
  fi
  n1="${NODES[1]}"
  c1=$(count_node "$n1")
  if [[ "$c0" -lt "$MAX_PER_NODE" && "$c0" -le "$c1" ]]; then
    echo "$n0" "$c0"
    return 0
  fi
  if [[ "$c1" -lt "$MAX_PER_NODE" ]]; then
    echo "$n1" "$c1"
    return 0
  fi
  if [[ "$c0" -lt "$MAX_PER_NODE" ]]; then
    echo "$n0" "$c0"
    return 0
  fi
  return 1
}

fill_slots() {
  local pending total c0 c1 node gpu dataset sam first
  pending=$(compute_pending_lines)
  c0=$(count_node "${NODES[0]}")
  if [[ "${#NODES[@]}" -ge 2 ]]; then
    c1=$(count_node "${NODES[1]}")
  else
    c1=0
  fi
  total=$((c0 + c1))
  log "status ${NODES[0]}=${c0} ${NODES[1]:-n/a}=${c1} total=${total} max_total=${MAX_TOTAL} pending_lines=$(echo "$pending" | grep -cve '^$' || true)"

  while true; do
    c0=$(count_node "${NODES[0]}")
    if [[ "${#NODES[@]}" -ge 2 ]]; then
      c1=$(count_node "${NODES[1]}")
    else
      c1=0
    fi
    total=$((c0 + c1))
    [[ "$total" -ge "$MAX_TOTAL" ]] && break
    pending=$(compute_pending_lines)
    [[ -z "${pending// }" ]] && break
    if ! read -r node gpu < <(pick_node_and_gpu); then
      break
    fi
    first=$(echo "$pending" | head -1)
    [[ -z "${first// }" ]] && break
    IFS=$'\t' read -r dataset sam <<<"$first"
    [[ -z "${dataset:-}" || -z "${sam:-}" ]] && break
    launch_on_node "$node" "$gpu" "$dataset" "$sam" || true
    # Allow remote Python to appear in /proc/<pid>/environ before next pending poll.
    sleep 15
  done
}

log "coordinator log=$COORD_LOG datasets=${QUEUE_DATASETS[*]} max_per_node=$MAX_PER_NODE max_total=$MAX_TOTAL mode=$MODE poll=${POLL_SEC}s nodes=${NODES[*]}"

while true; do
  fill_slots
  pending=$(compute_pending_lines)
  c0=$(count_node "${NODES[0]}")
  if [[ "${#NODES[@]}" -ge 2 ]]; then
    c1=$(count_node "${NODES[1]}")
  else
    c1=0
  fi
  total=$((c0 + c1))
  if [[ -z "${pending// }" && "$total" -eq 0 ]]; then
    log "nothing pending and nothing running — exiting."
    exit 0
  fi
  [[ "$MODE" == "once" ]] && exit 0
  sleep "$POLL_SEC"
done
