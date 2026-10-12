#!/usr/bin/env bash
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
# Build (once) and run the ByteTrack C++ runtime. The target must be
# explicit: host (detections replay, no SDK) or a board (video + model):
# x5, s100, s100p, s600. Each target builds in its own build/<target>/ dir.
#   ./run.sh --target x5 video --input track_test.mp4 --model-path yolov5n.bin --output out.mp4
#   ./run.sh --target s100 video --input track_test.mp4 --model-path yolov5x_672x672_nv12.hbm
#   ./run.sh --target host replay --detections stream.btcpp --records out.tsv
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
  echo "usage: run.sh --target host|x5|s100|s100p|s600 -- [bytetrack options]" >&2
  exit 2
}

if [[ "${1:-}" == "--target" ]]; then
  [[ $# -ge 2 && -n "${2:-}" ]] || usage
  TARGET=$2
  shift 2
else
  usage
fi
case "$TARGET" in
  host | x5 | s100 | s100p | s600) ;;
  *) echo "error: unknown target '$TARGET' (expected host, x5, s100, s100p or s600)" >&2; exit 2 ;;
esac
# The usage line documents a -- separator between launcher and tool options;
# consume it so the tool never sees it.
if [[ "${1:-}" == "--" ]]; then
  shift
fi

BUILD_DIR="$SCRIPT_DIR/build/$TARGET"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DBYTETRACK_TARGET="$TARGET" \
  -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIR" --target bytetrack -j >/dev/null

exec "$BUILD_DIR/bytetrack" "$@"
