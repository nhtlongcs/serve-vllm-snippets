#!/usr/bin/env bash
# Generic probe wrapper. Boots a model, walks image counts (1, 2, 4, ...) until
# OOM / context_overflow, then bisects to find the max images per chat request.
# Writes JSON to results/<config>-result.json and tees server logs to
# results/<config>-probe.log.
#
# Usage:
#   ./scripts/probe.sh <config-name> [extra args, e.g. --max-probe 128]
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

mkdir -p results
out_json="results/${config}-result.json"
out_log="results/${config}-probe.log"

uv run python test_max_images.py "$config_path" --out "$out_json" "$@" 2>&1 | tee "$out_log"
