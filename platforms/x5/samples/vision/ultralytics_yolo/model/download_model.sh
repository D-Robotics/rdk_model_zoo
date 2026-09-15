#!/usr/bin/env bash
# Download the default RDK X5 Ultralytics YOLO model assets.
#
# The maintained implementation lives in the canonical sample. This script
# forwards to it from the RDK X5 tree, keeping the documented command and the
# download location working. Model names and URLs come from the shared asset
# registry, so a downloaded file always matches the path the runtime resolves.
#
# Usage:
#   bash download_model.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SAMPLE_DIR=""
dir="${SCRIPT_DIR}"
while [ -n "${dir}" ] && [ "${dir}" != "/" ]; do
  if [ -f "${dir}/samples/vision/ultralytics_yolo/runtime/python/yolo_download.py" ]; then
    SAMPLE_DIR="${dir}/samples/vision/ultralytics_yolo"
    break
  fi
  dir="$(dirname "${dir}")"
done

if [ -z "${SAMPLE_DIR}" ]; then
  echo "[Error] the canonical Ultralytics YOLO sample was not found above ${SCRIPT_DIR}." >&2
  exit 1
fi

DOWNLOADER="${SAMPLE_DIR}/runtime/python/yolo_download.py"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[Error] python3 is required to resolve model assets." >&2
  exit 1
fi

exec python3 "${DOWNLOADER}" --platform x5 --model-dir "${SCRIPT_DIR}" "$@"
