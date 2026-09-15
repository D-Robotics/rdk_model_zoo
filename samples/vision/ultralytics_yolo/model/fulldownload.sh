#!/usr/bin/env bash
# Download every Ultralytics YOLO model asset a platform publishes.
#
# This is the full inventory: on RDK X5 it is the 67 published `.bin` models
# (YOLOv5u, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO12 and YOLOv13 across detect,
# segmentation, pose and classification); on the RDK S series it is every
# published `.hbm` model of the selected march.
#
# Usage:
#   bash fulldownload.sh                       # detected platform
#   bash fulldownload.sh --platform x5
#   bash fulldownload.sh --platform s600 --dry-run
#
# Prefer `download_model.sh` when only one model is needed.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/../runtime/python/yolo_download.py"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[Error] python3 is required to resolve model assets." >&2
  exit 1
fi

exec python3 "${PYTHON_SCRIPT}" --all "$@"
