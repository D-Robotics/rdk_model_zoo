#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/s600/diffusiondrive_r34_256x1024_s600.hbm}"
MODEL_URL="${MODEL_URL:-}"

if [ -f "${MODEL_PATH}" ]; then
  echo "Model already exists: ${MODEL_PATH}"
  exit 0
fi
if [ -z "${MODEL_URL}" ]; then
  echo "MODEL_URL is not set and the HBM model is absent: ${MODEL_PATH}" >&2
  echo "Copy the S600 HBM file to that path, or set MODEL_URL to an accessible download URL." >&2
  exit 1
fi

mkdir -p "$(dirname "${MODEL_PATH}")"
wget -c "${MODEL_URL}" -O "${MODEL_PATH}"
