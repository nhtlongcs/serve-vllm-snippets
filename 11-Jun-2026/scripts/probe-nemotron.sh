#!/usr/bin/env bash
# Probe Nemotron-3 Nano Omni 30B for max images per request.
set -e
exec "$(dirname "$0")/probe.sh" nemotron-omni-30b "$@"
