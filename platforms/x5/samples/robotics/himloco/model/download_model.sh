#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/bayes-e"
MODEL_NAME="himloco_go2_bayese_1x270.bin"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/himloco/${MODEL_NAME}"
EXPECTED_SHA256="7ce46ca2628f8bc236da0e8564180a1de92847bddf1ec00717ce7aa93e8c3e6a"

mkdir -p "${MODEL_DIR}"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"
if [[ ! -f "${MODEL_PATH}" ]]; then
  DOWNLOAD_PATH="$(mktemp "${MODEL_DIR}/.${MODEL_NAME}.XXXXXX")"
  trap 'rm -f "${DOWNLOAD_PATH}"' EXIT
  wget -O "${DOWNLOAD_PATH}" "${MODEL_URL}"
  mv "${DOWNLOAD_PATH}" "${MODEL_PATH}"
  trap - EXIT
fi

ACTUAL_SHA256="$(sha256sum "${MODEL_PATH}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  echo "SHA256 mismatch for ${MODEL_PATH}" >&2
  echo "expected ${EXPECTED_SHA256}" >&2
  echo "actual   ${ACTUAL_SHA256}" >&2
  exit 1
fi

echo "[INFO] Model ready: ${MODEL_PATH}"
