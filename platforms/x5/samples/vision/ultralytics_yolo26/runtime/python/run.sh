#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
while [ ! -f "$ROOT/samples/vision/ultralytics_yolo/runtime/python/yolo_dispatch.py" ]; do
  [ "$ROOT" != / ] || { echo 'Complete repository checkout required' >&2; exit 1; }
  ROOT="$(dirname "$ROOT")"
done
SAMPLE="$ROOT/samples/vision/ultralytics_yolo"
TASK="${1:-detect}"
case "$TASK" in -*) TASK=detect ;; *) shift || true ;; esac
exec python3 "$SCRIPT_DIR/main.py" --task "$TASK" "$@"
