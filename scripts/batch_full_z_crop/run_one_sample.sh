#!/usr/bin/env bash
# Run full-Z crop_cells for a single sample directory.
#
# Usage:
#   cd "$CMAP_ROOT" && bash scripts/batch_full_z_crop/run_one_sample.sh /abs/path/to/sample
#
# Environment:
#   CMAP_ROOT         repo root (auto-detected if unset)
#   CONDA_SH+CONDA_ENV or VENV_ACTIVATE  (Python environment)
#   CROP_FORCE        set to 1 to pass --force
#   MARGIN_XY / MARGIN_Z   override default margins

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAP_ROOT="${CMAP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export CMAP_ROOT

if [[ -f "$SCRIPT_DIR/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/env.sh"
fi

SAMPLE_DIR="${1:-}"
if [[ -z "$SAMPLE_DIR" ]]; then
  echo "Usage: $0 /absolute/path/to/sample_dir" >&2
  exit 1
fi
if [[ ! -d "$SAMPLE_DIR" ]]; then
  echo "ERROR: directory not found: $SAMPLE_DIR" >&2
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

EXTRA=()
if [[ "${CROP_FORCE:-0}" == "1" ]]; then
  EXTRA+=(--force)
fi
if [[ -n "${MARGIN_XY:-}" ]]; then
  EXTRA+=(--margin-xy "$MARGIN_XY")
fi
if [[ -n "${MARGIN_Z:-}" ]]; then
  EXTRA+=(--margin-z "$MARGIN_Z")
fi

exec python features/crop_cells.py \
  --data-dir "$SAMPLE_DIR" \
  --output-dir "$SAMPLE_DIR" \
  --also-full-z \
  "${EXTRA[@]}"
