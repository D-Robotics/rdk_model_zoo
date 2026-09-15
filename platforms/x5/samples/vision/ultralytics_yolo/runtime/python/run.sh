#!/usr/bin/env bash
# Run one Ultralytics YOLO task on RDK X5.
#
# The maintained implementation lives in the canonical sample. This script
# forwards to it from the RDK X5 tree, so the documented command keeps working
# without a second copy of the pipeline.
#
# Usage:
#   bash run.sh [task] [extra main.py options]
# The X5 tree only ships `.bin` artifacts, so the platform is pinned to x5
# here; pass --platform explicitly to override it.
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

has_platform=false
for arg in "$@"; do
  case "$arg" in --platform|--platform=*) has_platform=true ;; esac
done
if [ "$has_platform" = false ]; then
  set -- "$@" --platform x5
fi
exec bash "${SAMPLE_DIR}/runtime/python/run.sh" "$@"
