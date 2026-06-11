#!/usr/bin/env bash
# Generic serve wrapper. Picks a config from configs/ and starts vLLM via main.py.
#
# Usage:
#   ./scripts/serve.sh <config-name> [extra --flag overrides ...]
# Example:
#   ./scripts/serve.sh gemma4-31b --port 8001 --host 0.0.0.0
set -e
config="$1"
if [ -z "$config" ]; then
    echo "Usage: $0 <config-name> [extra args]" >&2
    echo "Available configs:" >&2
    ls "$(dirname "$0")/../configs"/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//' | sed 's/^/  /' >&2
    exit 2
fi
shift

# shellcheck disable=SC1091
source "$(dirname "$0")/_activate.sh"

config_path="configs/${config}.json"
if [ ! -f "$config_path" ]; then
    echo "ERROR: no config at $config_path" >&2
    exit 2
fi

exec uv run python main.py --config "$config_path" "$@"
