#!/usr/bin/env bash
# Serve Qwen3.6-27B on the default port (8000). Extra args pass through to main.py.
set -e
exec "$(dirname "$0")/serve.sh" qwen3_6-27b "$@"
