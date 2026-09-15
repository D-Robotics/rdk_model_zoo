#!/usr/bin/env bash
# Run one Ultralytics YOLO task on the RDK S series.
#
# The maintained implementation lives in the canonical sample. This script
# forwards to it from the RDK S tree, so the documented command keeps working
# without a second copy of the pipeline.
#
# Usage:
#   bash run.sh [task] [extra main.py options]
# S100, S100P and S600 publish different artifacts, so the platform is
# resolved from the board instead of being pinned here.
#
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

exec bash "${SAMPLE_DIR}/runtime/python/run.sh" "$@"
