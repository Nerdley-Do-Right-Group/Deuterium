#!/usr/bin/env bash
# Sets up the Deuterium Python virtual environment (.venv) on macOS and Linux.
# Usage: ./setup.sh

set -euo pipefail

# Always run from the repository root, even if invoked from elsewhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Deuterium Setup (macOS / Linux) ==="

# Prefer python3, fall back to python
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Error: Python 3.10+ was not found on PATH." >&2
    exit 1
fi

# Verify Python >= 3.10
PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Error: Python 3.10+ is required (found $PY_VERSION)." >&2
    exit 1
fi
echo "Using Python $PY_VERSION"

# Create .venv if it doesn't exist yet
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment at .venv ..."
    "$PYTHON_BIN" -m venv .venv
else
    echo "Virtual environment .venv already exists. Reusing it."
fi

VENV_PYTHON=".venv/bin/python"

echo "Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip

echo "Installing Deuterium and dependencies in editable mode..."
"$VENV_PYTHON" -m pip install -e ".[dev]"

echo ""
echo "Setup complete!"
echo "Next steps:"
echo "  1. Activate the venv:  source .venv/bin/activate"
echo "  2. Train the model:    deuterium train"
echo '  3. Predict a message:  deuterium predict "Congratulations! You won a free prize!"'
