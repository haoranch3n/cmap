#!/usr/bin/env bash
# Submit one LSF job per line: n_neighbors min_dist metric (CPU, standard queue).
#
#   export CMAP_ROOT=/abs/path/to/cmap
#   bash "$CMAP_ROOT/scripts/batch_dinov2_umap_sweep/generate_combo_lines.sh" > combos.txt
#   bash "$CMAP_ROOT/scripts/batch_dinov2_umap_sweep/submit_lsf.sh" combos.txt
#
# After all jobs finish, merge score JSONs:
#   python "$CMAP_ROOT/visualization/dinov2_umap_grid_search.py" --merge-scores \
#     --output-root "$CMAP_ROOT/output"
#
# Override queue / wall / memory (quote -R as a single argument):
#   export LSF_QUEUE=standard
#   export LSF_WALLTIME=120
#   export LSF_MEM_MB=65536

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export CMAP_ROOT

LIST="${1:-$SCRIPT_DIR/combos.txt}"
if [[ ! -f "$LIST" ]]; then
  echo "Usage: $0 /path/to/combos.txt  (each line: n_neighbors min_dist metric)" >&2
  exit 1
fi

LOGDIR="${LSF_LOG_DIR:-$CMAP_ROOT/scripts/batch_dinov2_umap_sweep/logs}"
mkdir -p "$LOGDIR"

LSF_QUEUE="${LSF_QUEUE:-standard}"
LSF_WALLTIME="${LSF_WALLTIME:-120}"
LSF_MEM_MB="${LSF_MEM_MB:-65536}"

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "${line// }" ]] && continue
  [[ "$line" =~ ^# ]] && continue
  read -r NN MD MET <<<"$line"
  if [[ -z "${NN:-}" || -z "${MD:-}" || -z "${MET:-}" ]]; then
    echo "Skip bad line: $line" >&2
    continue
  fi

  SAFE="nn${NN}_md${MD}_${MET}"
  SAFE="${SAFE//./p}"

  echo "bsub: n_neighbors=$NN min_dist=$MD metric=$MET"

  bsub \
    -J "dinov2_umap_${SAFE}" \
    -oo "${LOGDIR}/dinov2_umap_${SAFE}.out" \
    -eo "${LOGDIR}/dinov2_umap_${SAFE}.err" \
    -q "$LSF_QUEUE" \
    -W "$LSF_WALLTIME" \
    -n 1 \
    -R "rusage[mem=${LSF_MEM_MB}] span[hosts=1]" \
    bash "$SCRIPT_DIR/run_one_combo.sh" "$NN" "$MD" "$MET"

done < "$LIST"

echo "Submitted jobs for combos in $LIST"
echo "Logs: $LOGDIR"
echo "When finished: python visualization/dinov2_umap_grid_search.py --merge-scores --output-root $CMAP_ROOT/output"
