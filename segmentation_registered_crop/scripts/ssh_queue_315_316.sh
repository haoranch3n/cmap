#!/usr/bin/env bash
# Fair-queue daemon: run segmentation_registered_crop on nodegpu315 + nodegpu316
# via SSH + nohup. One cell per GPU slot, fills slots as workers finish.
#
# Adapted from scripts/no_deconv/ssh_queue_seg_registered_even_315_316.sh
#
# Usage (login node — run in screen/tmux or nohup it):
#   flock -n ~/cmap_crop_seg_queue.lock \
#     bash segmentation_registered_crop/scripts/ssh_queue_315_316.sh
#
#   CMAP_QUEUE_MODE=once bash segmentation_registered_crop/scripts/ssh_queue_315_316.sh
#
# Env overrides:
#   CMAP_QUEUE_NODES          (default: "nodegpu315 nodegpu316")
#   CMAP_QUEUE_MAX_PER_NODE   (default: 8 — one per GPU)
#   CMAP_QUEUE_POLL_SEC       (default: 60)
#   CMAP_QUEUE_MODE           daemon | once  (default: daemon)
#   CMAP_QUEUE_LOG_DIR

set -euo pipefail

PROJECT_ROOT="/research_jude/rgs01_jude/dept/DNB/core_operations/ImageAnalysis/Core/Haoran/cmap"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

CROPS_ROOT="/research/dept/dnb/core_operations/ImageAnalysisScratch/Gutierrez/CMAP_general/No_decon_tests/outputs/cell_crops/batch_processing_fullZ"
OUTPUT_ROOT="${PROJECT_ROOT}/output_registered_crop_seg"
SCRIPT="${PROJECT_ROOT}/segmentation_registered_crop/run_pipeline.py"

read -r -a NODES <<< "${CMAP_QUEUE_NODES:-nodegpu315 nodegpu316}"
MAX_PER_NODE="${CMAP_QUEUE_MAX_PER_NODE:-8}"
MAX_TOTAL=$(( ${#NODES[@]} * MAX_PER_NODE ))
POLL_SEC="${CMAP_QUEUE_POLL_SEC:-60}"
MODE="${CMAP_QUEUE_MODE:-daemon}"

LOG_DIR="${CMAP_QUEUE_LOG_DIR:-${PROJECT_ROOT}/logs/crop_seg_ssh_queue_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
COORD_LOG="${LOG_DIR}/coordinator.log"

log() { echo "$(date -Is) $*" | tee -a "$COORD_LOG"; }

# ── Discovery ─────────────────────────────────────────────────────────────────
# Emit "batch<TAB>sample_pos<TAB>cell_id" for every incomplete cell.
discover_incomplete() {
    python3 - "$CROPS_ROOT" "$OUTPUT_ROOT" <<'PY'
import sys
from pathlib import Path

crops_root = Path(sys.argv[1])
output_root = Path(sys.argv[2])
EXCLUDE = {"flagged_rotation_increase_gt5deg"}
CHANNELS = ["ch642", "ch488", "ch560"]

for tif in sorted(crops_root.glob("**/*_registered.tif")):
    parts = tif.parts
    # Skip flagged
    if EXCLUDE.intersection(parts):
        continue
    # Expect: CROPS_ROOT / batch / sample_pos / cell_id / *_registered.tif
    try:
        idx = parts.index(crops_root.name)
    except ValueError:
        continue
    rel = parts[idx+1:]
    if len(rel) < 3:
        continue
    batch, sample_pos, cell_id = rel[0], rel[1], rel[2]
    out_dir = output_root / batch / sample_pos / cell_id
    if all((out_dir / f"{cell_id}_{ch}_segmented.tif").exists() for ch in CHANNELS):
        continue
    print(f"{batch}\t{sample_pos}\t{cell_id}")
PY
}

# ── Busy detection ─────────────────────────────────────────────────────────────
# Returns "batch<TAB>sample_pos<TAB>cell_id" for running workers on all nodes.
ssh_running_cells() {
    for n in "${NODES[@]}"; do
        ssh -o BatchMode=yes -o ConnectTimeout=15 "$n" \
            env OUT="${OUTPUT_ROOT}" bash -s <<'EOSH' 2>/dev/null || true
for pid in $(pgrep -f 'segmentation_registered_crop/run_pipeline.py' 2>/dev/null || true); do
    [[ -r "/proc/${pid}/cmdline" ]] || continue
    cmd=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null) || continue
    # Extract --batch and --crop from cmdline
    batch=$(echo "$cmd" | grep -oP '(?<=--batch )\S+' || true)
    crop=$(echo "$cmd" | grep -oP '(?<=--crop )\S+' || true)
    [[ -z "$batch" || -z "$crop" ]] && continue
    # Find sample_pos by scanning output dir
    sample_pos=$(find "${OUT}/${batch}" -maxdepth 1 -type d -name "*" 2>/dev/null | \
        xargs -I{} sh -c 'test -d "{}/'"$crop"'" && basename {}' 2>/dev/null | head -1 || true)
    [[ -z "$sample_pos" ]] && sample_pos="unknown"
    printf '%s\t%s\t%s\n' "$batch" "$sample_pos" "$crop"
done
EOSH
    done | sort -u
}

# Count running workers on one node.
# Uses ps -eo cmd and anchors on '^python' to count only the Python interpreter
# processes, avoiding double-counting the bash -lc wrapper that also contains
# 'run_pipeline.py' in its argument string.
count_node() {
    local n="$1"
    local c
    c=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$n" \
        "ps -eo cmd 2>/dev/null | grep -c '^python.*segmentation_registered_crop/run_pipeline.py'" 2>/dev/null || echo 0) || c=0
    c=$(echo "$c" | tr -d '[:space:]')
    [[ "$c" =~ ^[0-9]+$ ]] || c=0
    echo "$c"
}

# Pending = incomplete − running.
compute_pending() {
    local disc busy
    disc=$(discover_incomplete | sort -u)
    busy=$(ssh_running_cells | awk '{print $1"\t"$3}' | sort -u)  # batch+cell_id
    # Filter out busy cells from disc
    while IFS=$'\t' read -r batch sample_pos cell_id; do
        key="${batch}	${cell_id}"
        if ! echo "$busy" | grep -qF "$key"; then
            printf '%s\t%s\t%s\n' "$batch" "$sample_pos" "$cell_id"
        fi
    done <<< "$disc"
}

# ── Slot selection ─────────────────────────────────────────────────────────────
pick_node_and_gpu() {
    local c0 c1 n0="${NODES[0]}"
    c0=$(count_node "$n0")
    if [[ "${#NODES[@]}" -lt 2 ]]; then
        if [[ "$c0" -lt "$MAX_PER_NODE" ]]; then echo "$n0 $c0"; return 0; fi
        return 1
    fi
    local n1="${NODES[1]}"
    c1=$(count_node "$n1")
    if [[ "$c0" -le "$c1" && "$c0" -lt "$MAX_PER_NODE" ]]; then echo "$n0 $c0"; return 0; fi
    if [[ "$c1" -lt "$MAX_PER_NODE" ]]; then echo "$n1 $c1"; return 0; fi
    if [[ "$c0" -lt "$MAX_PER_NODE" ]]; then echo "$n0 $c0"; return 0; fi
    return 1
}

# ── Launch ─────────────────────────────────────────────────────────────────────
launch() {
    local node="$1" gpu="$2" batch="$3" sample_pos="$4" cell_id="$5"
    local out="${LOG_DIR}/${node}__${batch}__${cell_id}_gpu${gpu}.out"
    local remote_cmd
    remote_cmd="
        eval \"\$(conda shell.bash hook)\"
        conda activate cmap
        ulimit -n 1048576 2>/dev/null || true
        export CUDA_VISIBLE_DEVICES=${gpu}
        python ${SCRIPT} \\
            --crops-root '${CROPS_ROOT}' \\
            --output-root '${OUTPUT_ROOT}' \\
            --batch '${batch}' \\
            --crop '${cell_id}'
    "
    log "START $node GPU=$gpu ${batch}/${cell_id} -> $out"
    ssh -o BatchMode=yes -o ConnectTimeout=30 "$node" \
        "nohup bash -lc $(printf %q "$remote_cmd") >>$(printf %q "$out") 2>&1 </dev/null &"
}

# ── Main fill loop ─────────────────────────────────────────────────────────────
fill_slots() {
    local c0=0 c1=0 total
    c0=$(count_node "${NODES[0]}")
    [[ "${#NODES[@]}" -ge 2 ]] && c1=$(count_node "${NODES[1]}") || c1=0
    total=$((c0 + c1))

    local pending
    pending=$(compute_pending)
    local n_pending
    n_pending=$(echo "$pending" | grep -vc '^$' || true)
    log "status ${NODES[0]}=${c0} ${NODES[1]:-n/a}=${c1} total=${total}/${MAX_TOTAL} pending=${n_pending}"

    while [[ -n "${pending// /}" ]]; do
        c0=$(count_node "${NODES[0]}")
        [[ "${#NODES[@]}" -ge 2 ]] && c1=$(count_node "${NODES[1]}") || c1=0
        total=$((c0 + c1))
        [[ "$total" -ge "$MAX_TOTAL" ]] && break

        local node gpu
        if ! read -r node gpu < <(pick_node_and_gpu); then break; fi

        local first
        first=$(echo "$pending" | head -1)
        [[ -z "${first// }" ]] && break

        local batch sample_pos cell_id
        IFS=$'\t' read -r batch sample_pos cell_id <<< "$first"
        [[ -z "${batch:-}" || -z "${cell_id:-}" ]] && break

        launch "$node" "$gpu" "$batch" "$sample_pos" "$cell_id" || true

        # Remove launched item from pending list
        pending=$(echo "$pending" | tail -n +2)

        sleep 10  # let the process appear in pgrep before next count
    done
}

# ── Entry ──────────────────────────────────────────────────────────────────────
log "Queue daemon starting. nodes=${NODES[*]} max_per_node=$MAX_PER_NODE max_total=$MAX_TOTAL poll=${POLL_SEC}s mode=$MODE"
log "crops_root=$CROPS_ROOT  output_root=$OUTPUT_ROOT"

while true; do
    fill_slots

    local_pending=$(compute_pending | grep -vc '^$' || true)
    c0=$(count_node "${NODES[0]}")
    [[ "${#NODES[@]}" -ge 2 ]] && c1=$(count_node "${NODES[1]}") || c1=0
    total=$((c0 + c1))

    if [[ "$local_pending" -eq 0 && "$total" -eq 0 ]]; then
        log "All cells complete — exiting."
        exit 0
    fi

    [[ "$MODE" == "once" ]] && exit 0
    sleep "$POLL_SEC"
done
