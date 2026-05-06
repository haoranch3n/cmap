# Copy to env.sh and customize. Sourced by run_one_sample.sh / submit scripts.
#
#   cp env.example.sh env.sh && vim env.sh

# Repo root (parent of features/, pipelines/, …)
export CMAP_ROOT="${CMAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Where pipeline artifacts live (must contain <sample>/cell_boxing/)
export PIPELINE_OUTPUT_DIR="${PIPELINE_OUTPUT_DIR:-$CMAP_ROOT/output}"

# --- Pick ONE way to get Python + torch ---

# Option A: conda
# export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
# export CONDA_ENV="cmap-torch"

# Option B: venv
# export VENV_ACTIVATE="/path/to/venv/bin/activate"

# Option C: environment modules (site-specific)
# module load cuda/12.1
# module load python/3.11

# Extra flags passed to extract_dinov2_embeddings.py for every sample
# export DINOV2_EXTRA_FLAGS="--apply-mask --batch-size 16"
