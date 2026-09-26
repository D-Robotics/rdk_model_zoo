#!/usr/bin/env bash
# Compatibility spelling; preparation remains explicit and target-scoped.
set -euo pipefail
exec bash "$(dirname -- "$0")/download.sh" "$@"
