#!/usr/bin/env bash
# Run one Ultralytics YOLO task on the detected board.
#
# Usage:
#   bash run.sh [task] [extra main.py options]
#
# Examples:
#   bash run.sh
#   bash run.sh detect
#   bash run.sh seg --family yolov8 --platform s100
#   bash run.sh cls --model-path ../../model/nash-e/yolo11n_cls_nashe_224x224_nv12.hbm
#
# The platform is detected from /sys/class/boardinfo unless --platform is given.
# No Python package is installed by this script: the required packages are part
# of the RDK system image. Install missing packages yourself if the board image
# does not provide them.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/main.py"

TASK="${1:-detect}"
case "${TASK}" in
  -*) TASK="detect" ;;
  *) shift || true ;;
esac

if ! command -v python3 >/dev/null 2>&1; then
  echo "[Error] python3 is required." >&2
  exit 1
fi

# main.py downloads the resolved default model when it is missing, and never
# downloads a model that was named explicitly with --model-path.
exec python3 "${MAIN_SCRIPT}" --task "${TASK}" "$@"
