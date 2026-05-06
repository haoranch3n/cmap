#!/usr/bin/env bash
# Run DINOv2 extraction for a single sample (--data-rel <name>).
# Usage:
#   cd "$CMAP_ROOT" && bash scripts/batch_dinov2/run_one_sample.sh SAMPLE_NAME
#
# Environment:
#   CMAP_ROOT, PIPELINE_OUTPUT_DIR, CONDA_SH+CONDA_ENV or VENV_ACTIVATE,
#   DINOV2_EXTRA_FLAGS (optional, appended to python argv)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export CMAP_ROOT

if [[ -f "$SCRIPT_DIR/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/env.sh"
fi

SAMPLE="${1:-}"
if [[ -z "$SAMPLE" ]]; then
  echo "Usage: $0 <sample_name>" >&2
  echo "  sample_name = directory under PIPELINE_OUTPUT_DIR (same as --data-rel)" >&2
  exit 1
fi

if [[ -n "${CONDA_SH:-}" && -n "${CONDA_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
elif [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
fi

cd "$CMAP_ROOT"
export PYTHONUNBUFFERED=1

# Space-separated extra CLI flags, e.g. DINOV2_EXTRA_FLAGS="--apply-mask --force"
read -r -a EXTRA <<< "${DINOV2_EXTRA_FLAGS:-}"

exec python features/extract_dinov2_embeddings.py \
  --data-rel "$SAMPLE" \
  --device "${DINOV2_DEVICE:-cuda}" \
  "${EXTRA[@]}"
