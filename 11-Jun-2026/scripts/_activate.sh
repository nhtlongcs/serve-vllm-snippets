# Source-only helper. Activates the run-llm conda env (CUDA toolkit + libstdc++)
# and chdir's to the project root, so callers can `uv run python ...` and have
# CONDA_PREFIX wired up — main.py reads it to set LIBRARY_PATH / LD_LIBRARY_PATH.
#
# Usage from a wrapper script:
#     source "$(dirname "$0")/_activate.sh"

# `set -u` breaks conda's activate scripts (they reference unbound CONDA_BACKUP_*),
# so callers should run with `set -e` only.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "ERROR: conda not found at $CONDA_ROOT. Set CONDA_ROOT to your conda install." >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

if ! conda env list | awk 'NF>0 && $1!~/^#/{print $1}' | grep -qx run-llm; then
    echo "ERROR: conda env 'run-llm' not found. Run ./setup_env.sh first." >&2
    return 1 2>/dev/null || exit 1
fi

conda activate run-llm
