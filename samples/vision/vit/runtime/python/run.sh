#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ "${1:-}" == "int8" || "${1:-}" == "int16" ]]; then
  variant="$1"
  shift
  set -- --variant "$variant" "$@"
fi
exec python3 "$SCRIPT_DIR/main.py" "$@"
