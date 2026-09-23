#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-s100}"
VARIANT="${2:-}"

if [[ -n "$VARIANT" ]]; then
  exec python3 "${SCRIPT_DIR}/download.py" --target "${TARGET}" --variant "${VARIANT}" --output-dir "${SCRIPT_DIR}"
fi
exec python3 "${SCRIPT_DIR}/download.py" --target "${TARGET}" --output-dir "${SCRIPT_DIR}"
