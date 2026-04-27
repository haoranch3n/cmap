#!/bin/bash
# Fix volumina for Python 3.9 compatibility
# Adds "from __future__ import annotations" to all .py files that use
# PEP 604 union type hints (e.g. float | int) which require Python 3.10+.
# The __future__ import makes these annotations strings, avoiding the error.
#
# Usage: bash fix_volumina.sh [env_name]
#   Default env_name: iglusnfr_processing

ENV_NAME="${1:-iglusnfr_processing}"
CONDA_ENVS="$HOME/.conda/envs"
SITE_PKGS="$CONDA_ENVS/$ENV_NAME/lib/python3.9/site-packages/volumina"

if [ ! -d "$SITE_PKGS" ]; then
    echo "ERROR: volumina not found at: $SITE_PKGS"
    echo "Check that the environment '$ENV_NAME' exists and has volumina installed."
    exit 1
fi

PATCHED=0

# Find all .py files containing PEP 604 type union syntax (X | Y)
# that are missing the __future__ annotations import
for pyfile in $(grep -rl ' | ' "$SITE_PKGS" --include='*.py' 2>/dev/null); do
    # Check if file actually has PEP 604 type hints (not just bitwise or)
    if ! grep -qP '\b(float|int|bool|str|None)\s*\|\s*(float|int|bool|str|None)' "$pyfile" 2>/dev/null; then
        continue
    fi
    # Skip if already patched
    if head -5 "$pyfile" | grep -q "from __future__ import annotations"; then
        continue
    fi
    # Patch: prepend the import
    TMPFILE=$(mktemp)
    echo 'from __future__ import annotations' > "$TMPFILE"
    cat "$pyfile" >> "$TMPFILE"
    cp "$TMPFILE" "$pyfile"
    rm "$TMPFILE"
    echo "  Patched: $(basename "$pyfile")"
    PATCHED=$((PATCHED + 1))
done

# Clear .pyc caches so Python picks up the patched files
find "$SITE_PKGS" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

if [ "$PATCHED" -eq 0 ]; then
    echo "No files needed patching (already fixed or no PEP 604 syntax found)."
else
    echo ""
    echo "Patched $PATCHED file(s). Cleared __pycache__."
fi

# Verify
echo ""
echo "Verifying..."
"$CONDA_ENVS/$ENV_NAME/bin/python" -c "import volumina; print('volumina v' + volumina.__version__ + ' - OK')" 2>&1
