#!/usr/bin/env bash
# Probe Gemma 4 31B for max images per request.
set -e
exec "$(dirname "$0")/probe.sh" gemma4-31b "$@"
