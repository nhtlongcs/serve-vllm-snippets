#!/usr/bin/env bash
# Bootstrap the run-llm environment: mamba env (CUDA toolkit) + uv .venv (python deps).
# Idempotent: rerun any time env.yml or pyproject.toml change.

set -eo pipefail
# Conda's activate/deactivate scripts trip on `set -u` (they reference
# unbound CONDA_BACKUP_* vars), so we intentionally leave nounset off.

cd "$(dirname "$0")"

CONDA_BIN="${CONDA_BIN:-}"
if [ -z "$CONDA_BIN" ]; then
    if command -v mamba >/dev/null 2>&1; then
        CONDA_BIN=mamba
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BIN=conda
    else
        echo "ERROR: mamba (preferred) or conda is required on PATH" >&2
        exit 1
    fi
fi

ENV_NAME=$(awk '/^name:/{print $2; exit}' env.yml)
if [ -z "$ENV_NAME" ]; then
    echo "ERROR: could not read env name from env.yml" >&2
    exit 1
fi

if "$CONDA_BIN" env list 2>/dev/null | awk 'NF>0 && $1!~/^#/{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> Updating conda env '$ENV_NAME' from env.yml"
    "$CONDA_BIN" env update -n "$ENV_NAME" -f env.yml --prune
else
    echo ">>> Creating conda env '$ENV_NAME' from env.yml (one-time CUDA toolkit download)"
    "$CONDA_BIN" env create -f env.yml
fi

# Activate the env in this shell so uv picks up the conda python & toolchain.
# mamba 1.x doesn't implement `shell.bash hook`; fall back to conda for it.
HOOK_BIN=$CONDA_BIN
if [ "$CONDA_BIN" = "mamba" ] && command -v conda >/dev/null 2>&1; then
    HOOK_BIN=conda
fi
eval "$("$HOOK_BIN" shell.bash hook)"
conda activate "$ENV_NAME"

echo ">>> CONDA_PREFIX=$CONDA_PREFIX"
echo ">>> python=$(command -v python)  uv=$(command -v uv)"

# If a previous .venv was created against a Python outside this conda env, drop
# it so uv recreates against the conda interpreter (keeps libstdc++/CUDA stack
# coherent).
if [ -d .venv ]; then
    venv_python_real=$(readlink -f .venv/bin/python || true)
    if [ -n "$venv_python_real" ] && [[ "$venv_python_real" != "$CONDA_PREFIX"* ]]; then
        echo ">>> Removing stale .venv (built against $venv_python_real)"
        rm -rf .venv
    fi
fi

echo ">>> uv sync"
uv sync

cat <<MSG

Setup complete.
  conda env  : $ENV_NAME  ($CONDA_PREFIX)
  python venv: $(realpath .venv 2>/dev/null || echo .venv)

Activate the env in a new shell with:
  $CONDA_BIN activate $ENV_NAME

Then run anything via uv (main.py and test_max_images.py pick CUDA stubs up
from \$CONDA_PREFIX automatically):
  uv run python main.py --config configs/gemma4-31b.json
  uv run python test_max_images.py configs/gemma4-31b.json --max-probe 64
MSG
