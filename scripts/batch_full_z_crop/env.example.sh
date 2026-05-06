# Copy to env.sh and customize. Sourced by run_one_sample.sh / submit scripts.
#
#   cp env.example.sh env.sh && vim env.sh

# Repo root (parent of features/, pipelines/, …)
export CMAP_ROOT="${CMAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Where pipeline artifacts live (must contain <sample>/filtered_642_combined.tif)
export PIPELINE_OUTPUT_DIR="${PIPELINE_OUTPUT_DIR:-$CMAP_ROOT/output}"

# --- Pick ONE way to get Python ---

# Option A: conda
# export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
# export CONDA_ENV="cmap"

# Option B: venv
# export VENV_ACTIVATE="/path/to/venv/bin/activate"

# Option C: environment modules (site-specific)
# module load python/3.11

# --- Crop options ---

# Pass --force to overwrite existing crops
# export CROP_FORCE=1

# Override default XY / Z margins
# export MARGIN_XY=20
# export MARGIN_Z=5
