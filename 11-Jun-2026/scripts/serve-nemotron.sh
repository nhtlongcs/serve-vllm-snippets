#!/usr/bin/env bash
# Serve Nemotron-3 Nano Omni 30B-A3B Reasoning on the default port (8000).
# Extra args pass through to main.py.
set -e
exec "$(dirname "$0")/serve.sh" nemotron-omni-30b "$@"
