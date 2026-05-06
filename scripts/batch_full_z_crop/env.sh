export CMAP_ROOT="${CMAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PIPELINE_OUTPUT_DIR="${PIPELINE_OUTPUT_DIR:-$CMAP_ROOT/output}"

export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_ENV="cmap"
