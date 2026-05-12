#!/usr/bin/env bash
# Worker: Cellpose segmentation for one sample (488 / 560 / 642) from Gutierrez
# registered TIFFs. Runs native segmentation/run_pipeline.py three times (one
# channel per run) so outputs match postprocess layout:
#   $NO_DECONV_MERGE/488nm_crop/segmentation_3D_masks/488nm_crop_3D_indexed.tif
#
# Required environment (absolute paths):
#   NO_DECONV_REGISTERED — dir with 488nm_registered.tif, 560nm_registered.tif,
#                          642nm_reference.tif
#   NO_DECONV_MERGE      — per-sample merge root under output_no_deconv/...
#   PROJECT_ROOT         — cmap repo root (contains segmentation/)
#
# Reads only from NO_DECONV_REGISTERED; writes only under NO_DECONV_MERGE.
# No LSF email flags.

set -euo pipefail

: "${NO_DECONV_REGISTERED:?}"
: "${NO_DECONV_MERGE:?}"
: "${PROJECT_ROOT:?}"

REG="$(cd "$NO_DECONV_REGISTERED" && pwd)"
MER="$(cd "$NO_DECONV_MERGE" && pwd)"
ROOT="$(cd "$PROJECT_ROOT" && pwd)"

F488="${REG}/488nm_registered.tif"
F560="${REG}/560nm_registered.tif"
F642="${REG}/642nm_reference.tif"
for f in "$F488" "$F560" "$F642"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing required input: $f" >&2
    exit 1
  fi
done

mkdir -p "$MER"

# Symlinks at merge root so postprocess can resolve originals (488nm_crop.tif).
ln -sfn "$F488" "${MER}/488nm_crop.tif"
ln -sfn "$F560" "${MER}/560nm_crop.tif"
ln -sfn "$F642" "${MER}/642nm_crop.tif"

RUN_PY="${ROOT}/segmentation/run_pipeline.py"
if [[ ! -f "$RUN_PY" ]]; then
  echo "ERROR: missing $RUN_PY" >&2
  exit 1
fi

log="${MER}/segmentation_cellpose_registered.log"
{
  echo "=== $(date -Is) no-deconv segmentation (3 channels) ==="
  echo "REG=$REG"
  echo "MER=$MER"
  echo "PROJECT_ROOT=$ROOT"
  nvidia-smi 2>&1 || true
  python3 -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())" 2>&1 || true
} | tee -a "$log"

run_channel() {
  local stem="$1" src="$2"
  local tmp
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap 'rm -rf "$tmp"' RETURN
  ln -sf "$src" "${tmp}/${stem}.tif"
  export PIPELINE_DATA_DIR="$tmp"
  export PIPELINE_OUTPUT_DIR="${MER}/${stem}"
  mkdir -p "$PIPELINE_OUTPUT_DIR"
  echo "" | tee -a "$log"
  echo ">>> $(date -Is) channel ${stem} PIPELINE_OUTPUT_DIR=${PIPELINE_OUTPUT_DIR}" | tee -a "$log"
  (cd "$ROOT" && exec python3 -u "$RUN_PY") 2>&1 | tee -a "$log"
}

run_channel "488nm_crop" "$F488"
run_channel "560nm_crop" "$F560"
run_channel "642nm_crop" "$F642"

echo "=== $(date -Is) done ===" | tee -a "$log"
