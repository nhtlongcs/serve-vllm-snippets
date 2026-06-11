#!/usr/bin/env bash
# Probe Qwen3.6-27B for max images per request.
set -e
exec "$(dirname "$0")/probe.sh" qwen3_6-27b "$@"
