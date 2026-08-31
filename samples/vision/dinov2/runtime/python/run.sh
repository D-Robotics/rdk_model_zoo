#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}"

OUTPUT=${1:-cls_feat}
shift || true

bash ../../model/download_model.sh

python3 main.py \
  --output "${OUTPUT}" \
  "$@"
