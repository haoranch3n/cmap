#!/bin/bash
# CMAP Cell Exemption Plugin - Launcher
#
# Mirrors napari-plugin/run.sh: load the napari module (which provides a working
# conda + napari + Qt + OpenGL stack), make sure the plugin is installed, then
# launch napari with the CMAP Cell Exemption widget docked.
#
# Usage:
#   ./run.sh                                   # open the 95-sample symlink farm (real_data/)
#   ./run.sh /path/to/output_root              # open and auto-discover samples under that root
#   ./run.sh /path/to/output_root --annotation-dir /path/to/annotations
#   ./run.sh --check                           # environment diagnostics, no GUI
#
# By default (no positional path) this opens the symlink farm built by
# setup_real_symlinks.sh, which links the 95 real combined TIFFs + coordinate
# CSVs without duplicating storage. Rebuild it any time with:
#   bash setup_real_symlinks.sh
#
# The default CSV output folder is annotations/<reviewer>/<date>/ (override with
# --annotation-dir /some/path).

# Prefer logical cwd so symlink/nfs paths stay usable from the GUI
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -L 2>/dev/null || pwd)"

# Default data root = the symlink farm (override by passing a path as $1).
DEFAULT_ROOT="$SCRIPT_DIR/real_data"
HAS_POSITIONAL=0
skip_next=0
for arg in "$@"; do
    if [[ "$skip_next" -eq 1 ]]; then skip_next=0; continue; fi
    case "$arg" in
        --annotation-dir) skip_next=1 ;;   # its value is not a positional root
        --annotation-dir=*) ;;
        -*) ;;                             # other flags like --check
        *) HAS_POSITIONAL=1; break ;;
    esac
done
if [[ "$HAS_POSITIONAL" -eq 0 ]] && [[ "$*" != *"--check"* ]] && [[ -d "$DEFAULT_ROOT" ]]; then
    set -- "$DEFAULT_ROOT" "$@"
fi

# Default CSV output folder = annotations/ (the plugin files sessions under
# <reviewer>/<date>/ inside it). Override with --annotation-dir.
DEFAULT_ANN="$SCRIPT_DIR/annotations"
if [[ "$*" != *"--check"* ]] && [[ "$*" != *"--annotation-dir"* ]]; then
    set -- "$@" --annotation-dir "$DEFAULT_ANN"
fi

# Load the napari module (provides conda + napari + qtpy + a working OpenGL stack).
# This is the key difference from running conda-base python directly, which can
# fail to load the system Mesa driver (MESA-LOADER / swrast / QOpenGLWidget errors).
if command -v module &>/dev/null; then
    module load napari 2>/dev/null
fi

# Verify napari is available
if ! python -c "import napari" 2>/dev/null; then
    echo "ERROR: napari not found."
    echo ""
    echo "Try:"
    echo "  module load napari"
    echo "  ./run.sh"
    exit 1
fi

# Ensure the plugin is installed for the module's python
if ! python -c "import cmap_cell_exemption_plugin" 2>/dev/null; then
    echo "Installing cmap-cell-exemption-plugin..."
    pip install --user -e "$SCRIPT_DIR" --quiet
fi

exec python -m cmap_cell_exemption_plugin "$@"
