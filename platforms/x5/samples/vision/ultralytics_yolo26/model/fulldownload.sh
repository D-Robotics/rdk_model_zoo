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
exec python3 "$SAMPLE/runtime/python/yolo_download.py" --family yolo26 --all --model-dir "$SCRIPT_DIR" "$@"
