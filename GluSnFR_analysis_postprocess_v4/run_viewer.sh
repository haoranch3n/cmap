#!/bin/bash
# iGluSnFR Analysis Toolbox v3 - Viewer Launcher
#
# Usage:
#   ./run_viewer.sh                   # auto-detect env
#   ./run_viewer.sh --env myenvname   # use specific env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENVS="$HOME/.conda/envs"

# Check for --env argument
if [[ "${1:-}" == "--env" && -n "${2:-}" ]]; then
    EXPLICIT_ENV="$2"
    shift 2
    if [ -f "$CONDA_ENVS/$EXPLICIT_ENV/bin/python" ]; then
        echo "Using viewer environment: $EXPLICIT_ENV"
        cd "$SCRIPT_DIR"
        exec "$CONDA_ENVS/$EXPLICIT_ENV/bin/python" viewer.py "$@"
    else
        echo "ERROR: Environment '$EXPLICIT_ENV' not found at $CONDA_ENVS/$EXPLICIT_ENV"
        exit 1
    fi
fi

# Auto-detect: try environments in order of preference
for env in iglusnfr_viewer napari_viewer; do
    if [ -f "$CONDA_ENVS/$env/bin/python" ]; then
        echo "Using viewer environment: $env"
        cd "$SCRIPT_DIR"
        exec "$CONDA_ENVS/$env/bin/python" viewer.py "$@"
    fi
done

echo "ERROR: No suitable viewer conda environment found."
echo ""
echo "Available options:"
echo "  1. Run setup.sh to create the default environment:"
echo "       chmod +x setup.sh && ./setup.sh"
echo ""
echo "  2. Specify an existing environment by name:"
echo "       ./run_viewer.sh --env your_env_name"
echo ""
echo "  3. Create manually:"
echo "       conda env create -f environment_viewer.yml"
echo "       ./run_viewer.sh"
exit 1
