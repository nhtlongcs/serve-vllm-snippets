#!/usr/bin/env bash
# Serve Gemma 4 31B Instruct on the default port (8000). Extra args pass through
# to main.py (e.g. --port 8001, --host 127.0.0.1, or any vllm flag).
set -e
exec "$(dirname "$0")/serve.sh" gemma4-31b "$@"
