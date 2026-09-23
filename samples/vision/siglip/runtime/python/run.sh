#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ "${1:-}" == "pooler_output" || "${1:-}" == "last_hidden_state" ]]; then
  submodel="$1"
  shift
  set -- --submodel "$submodel" "$@"
fi
exec python3 "$SCRIPT_DIR/main.py" "$@"
