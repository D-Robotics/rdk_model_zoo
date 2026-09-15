#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
while [ ! -f "$ROOT/samples/vision/ultralytics_yolo/runtime/python/yolo_dispatch.py" ]; do
  [ "$ROOT" != / ] || { echo 'Complete repository checkout required' >&2; exit 1; }
  ROOT="$(dirname "$ROOT")"
done
SAMPLE="$ROOT/samples/vision/ultralytics_yolo"
set -- --platform x5 "$@"
# Old download command selects nano for all five tasks unless a task is explicit.
HAS_TASK=false
for arg in "$@"; do case "$arg" in --task|--task=*) HAS_TASK=true ;; esac; done
if [ "$HAS_TASK" = true ]; then
  exec python3 "$SAMPLE/runtime/python/yolo_download.py" --family yolo26 --model-dir "$SCRIPT_DIR" "$@"
fi
for task in detect seg pose cls obb; do
  python3 "$SAMPLE/runtime/python/yolo_download.py" --family yolo26 --task "$task" --model-dir "$SCRIPT_DIR" "$@"
done
