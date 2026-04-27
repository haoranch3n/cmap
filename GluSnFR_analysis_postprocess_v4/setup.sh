#!/bin/bash
# iGluSnFR Analysis Toolbox v3 - Setup Script
# Creates/configures conda environments for processing and viewing
#
# Usage:
#   ./setup.sh              # Safe: skips existing envs
#   ./setup.sh --force      # Prompts to recreate existing envs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENVS="$HOME/.conda/envs"
FORCE_RECREATE=false

if [[ "${1:-}" == "--force" ]]; then
    FORCE_RECREATE=true
fi

echo "========================================"
echo "iGluSnFR Analysis Toolbox v3 - Setup"
echo "========================================"
echo ""
echo "Environments directory: $CONDA_ENVS"
echo "Force recreate: $FORCE_RECREATE"
echo ""

# Load conda (try module system first, then fall back to conda init)
if command -v module &>/dev/null; then
    echo "Loading conda via module..."
    module load conda 2>/dev/null || true
fi

if ! command -v conda &>/dev/null; then
    echo "Trying conda from shell init..."
    eval "$(conda shell.bash hook 2>/dev/null)" || true
fi

if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Please install Miniconda/Anaconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "conda found: $(conda --version)"
echo ""

# ========================================
# NVIDIA DRIVER CHECK
# ========================================
echo "========================================"
echo "GPU / NVIDIA Driver Check"
echo "========================================"
if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VER=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU:    $GPU_NAME"
    echo "  Driver: $DRIVER_VER"

    # Parse major driver version
    DRIVER_MAJOR=$(echo "$DRIVER_VER" | cut -d. -f1)
    if [ "$DRIVER_MAJOR" -lt 525 ] 2>/dev/null; then
        echo ""
        echo "  WARNING: Driver $DRIVER_VER is below 525. CUDA 12.x requires >= 525."
        echo "  For Blackwell GPUs (B100/B200): driver >= 570 recommended."
        echo "  Please upgrade: https://www.nvidia.com/Download/index.aspx"
    elif [ "$DRIVER_MAJOR" -lt 570 ] 2>/dev/null; then
        echo "  NOTE: Driver $DRIVER_VER works with CUDA 12.x."
        echo "  For Blackwell GPUs: driver >= 570 recommended."
    else
        echo "  Driver OK for all supported GPUs including Blackwell."
    fi
else
    echo "  nvidia-smi not found. GPU acceleration will not be available."
fi
echo ""

# ========================================
# VIEWER ENVIRONMENT
# ========================================
echo "========================================"
echo "1. Viewer Environment Setup"
echo "========================================"

ENV_VIEWER="iglusnfr_viewer"

create_viewer_env() {
    local env_name=$1
    echo "Creating viewer environment: $env_name"
    echo "This will take 3-5 minutes..."
    echo ""

    # Create with ALL Qt packages from conda to avoid library conflicts
    # DO NOT install PyQt5/PyQtWebEngine via pip - causes symbol conflicts
    conda create -n "$env_name" -c conda-forge \
        python=3.10 \
        "napari>=0.5" \
        pyqt \
        pyqtwebengine \
        qt-main \
        plotly \
        psutil \
        pandas \
        scipy \
        scikit-image \
        tifffile \
        matplotlib \
        polars \
        pyarrow \
        -y

    echo "Viewer environment created: $env_name"
}

while [ -d "$CONDA_ENVS/$ENV_VIEWER" ]; do
    echo "Environment '$ENV_VIEWER' already exists."
    echo "  (conda envs in \$HOME are shared across servers)"
    echo ""
    if [ "$FORCE_RECREATE" = true ]; then
        read -p "  Remove and recreate '$ENV_VIEWER'? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            conda env remove -n "$ENV_VIEWER" -y
            break
        fi
    fi
    read -p "  Enter a different name (or press Enter to skip): " NEW_NAME
    if [ -z "$NEW_NAME" ]; then
        echo "  Skipping viewer environment creation."
        break
    fi
    ENV_VIEWER="$NEW_NAME"
done

if [ ! -d "$CONDA_ENVS/$ENV_VIEWER" ]; then
    create_viewer_env "$ENV_VIEWER"
fi

echo "Viewer environment: $ENV_VIEWER"
echo ""

# ========================================
# PROCESSING ENVIRONMENT (with ilastik)
# ========================================
echo "========================================"
echo "2. Processing Environment Setup"
echo "========================================"

ENV_PROC="iglusnfr_processing"

create_processing_env() {
    local env_name=$1

    echo ""
    echo "Creating processing environment: $env_name"
    echo "========================================"
    echo "This will take 5-15 minutes..."
    echo "========================================"
    echo ""

    # Create ilastik cache directory (fixes parallel logging race condition)
    mkdir -p "$HOME/.cache/ilastik/log"

    # Use yml file, override name with -n so user-chosen name is used
    conda env create -f "$SCRIPT_DIR/environment_processing.yml" -n "$env_name"

    echo ""
    echo "Processing environment created: $env_name"
}

while [ -d "$CONDA_ENVS/$ENV_PROC" ]; do
    echo "Environment '$ENV_PROC' already exists."
    echo "  (conda envs in \$HOME are shared across servers)"
    echo ""
    if [ "$FORCE_RECREATE" = true ]; then
        read -p "  Remove and recreate '$ENV_PROC'? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            conda env remove -n "$ENV_PROC" -y
            break
        fi
    fi
    read -p "  Enter a different name (or press Enter to skip): " NEW_NAME
    if [ -z "$NEW_NAME" ]; then
        echo "  Skipping processing environment creation."
        break
    fi
    ENV_PROC="$NEW_NAME"
done

if [ ! -d "$CONDA_ENVS/$ENV_PROC" ]; then
    create_processing_env "$ENV_PROC"
fi

echo ""

# ========================================
# VERIFY & PATCH
# ========================================
echo "========================================"
echo "Verifying installation..."
echo "========================================"

# Test viewer
echo -n "Viewer environment: "
if "$CONDA_ENVS/$ENV_VIEWER/bin/python" -c "import napari; import plotly; import polars; print('OK')" 2>/dev/null; then
    :
else
    echo "FAILED - some packages missing"
fi

# Test processing (core imports)
echo -n "Processing environment: "
if "$CONDA_ENVS/$ENV_PROC/bin/python" -c "import ilastik; import cupy; import numba; print('OK')" 2>/dev/null; then
    :
else
    echo "PARTIAL - some GPU packages may be missing (cucim optional)"
fi

# ---- volumina Python 3.9 compatibility check & auto-patch ----
# volumina 1.3.x may ship with PEP 604 type hints (float | int | bool)
# but without "from __future__ import annotations", which breaks on
# Python 3.9.  Detect and patch automatically.
echo -n "  volumina compatibility: "
if "$CONDA_ENVS/$ENV_PROC/bin/python" -c "import volumina; print('OK (v' + volumina.__version__ + ')')" 2>/dev/null; then
    :
else
    echo "FAILED - attempting auto-patch..."
    SITE_PKGS="$CONDA_ENVS/$ENV_PROC/lib/python3.9/site-packages/volumina"

    if [ -d "$SITE_PKGS" ]; then
        PATCHED=0
        # Find all .py files with PEP 604 type unions that lack the __future__ import
        while IFS= read -r pyfile; do
            # Verify it actually has PEP 604 type hints (not just bitwise or)
            if ! grep -qP '\b(float|int|bool|str|None)\s*\|\s*(float|int|bool|str|None)' "$pyfile" 2>/dev/null; then
                continue
            fi
            # Skip if already patched
            if head -5 "$pyfile" | grep -q "from __future__ import annotations"; then
                continue
            fi
            # Prepend the import
            TMPFILE=$(mktemp)
            echo 'from __future__ import annotations' > "$TMPFILE"
            cat "$pyfile" >> "$TMPFILE"
            cp "$TMPFILE" "$pyfile"
            rm "$TMPFILE"
            echo "    Patched: $pyfile"
            PATCHED=$((PATCHED + 1))
        done < <(grep -rl ' | ' "$SITE_PKGS" --include='*.py' 2>/dev/null)

        # Clear __pycache__ so Python picks up patched files
        find "$SITE_PKGS" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

        if [ "$PATCHED" -gt 0 ]; then
            echo "    Patched $PATCHED file(s), cleared __pycache__."
        fi

        # Re-verify
        echo -n "  volumina (after patch): "
        if "$CONDA_ENVS/$ENV_PROC/bin/python" -c "import volumina; print('OK (v' + volumina.__version__ + ')')" 2>/dev/null; then
            :
        else
            echo "STILL FAILED"
            echo "  Manual fix: bash $SCRIPT_DIR/fix_volumina.sh $ENV_PROC"
        fi
    else
        echo "  volumina not found at $SITE_PKGS"
    fi
fi

# ========================================
# SUMMARY
# ========================================
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Environments:"
echo "  Viewer:     $CONDA_ENVS/$ENV_VIEWER"
echo "  Processing: $CONDA_ENVS/$ENV_PROC"
echo ""
echo "To launch the viewer:"
echo "  ./run_viewer.sh"
echo ""
echo "Or directly:"
echo "  $CONDA_ENVS/$ENV_VIEWER/bin/python viewer.py"
echo ""
