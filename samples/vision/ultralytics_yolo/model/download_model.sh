#!/usr/bin/env bash
# Download the Ultralytics YOLO model assets published for a platform.
#
# Usage:
#   bash download_model.sh [options]
#   bash download_model.sh [soc] [family] [task] [model_size]   # legacy form
#
# Options:
#   --platform  x5 | s100 | s100p | s600   Target platform (default: detected board)
#   --family    yolo11 | yolov8 | ...      Model family (default: yolo11)
#   --task      detect | seg | pose | cls  Task (default: the platform default set)
#   --model-size n | s | m | l | x | ...   Model scale (default: family default)
#   --asset-id group:sample:filename      Exact standalone S source record
#   --all                                  Every asset the platform publishes
#   --dry-run                              Print the plan without downloading
#
# Examples:
#   bash download_model.sh --platform x5
#   bash download_model.sh --platform s600 --family yolov8 --task seg
#   bash download_model.sh s100 yolo11 detect n
#
# Model names and URLs are derived from the same asset registry the runtime
# uses, so the downloaded file always matches the path the sample resolves.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/../runtime/python/yolo_download.py"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[Error] python3 is required to resolve model assets." >&2
  exit 1
fi

exec python3 "${PYTHON_SCRIPT}" "$@"
