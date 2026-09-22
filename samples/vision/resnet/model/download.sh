#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-x5}"
VARIANT="${2:-resnet18}"

exec python3 "${SCRIPT_DIR}/download.py" --target "${TARGET}" --variant "${VARIANT}" --output-dir "${SCRIPT_DIR}"
