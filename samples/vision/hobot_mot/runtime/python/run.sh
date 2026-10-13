#!/usr/bin/env bash
# Locate the Python entry point and forward arguments and the exit code.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/main.py" "$@"
