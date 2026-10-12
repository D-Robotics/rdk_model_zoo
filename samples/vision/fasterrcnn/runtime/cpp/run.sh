#!/usr/bin/env bash
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
# Build (once) and run the Faster R-CNN C++ runtime. The target must be
# explicit: host (model-free modes only: --help/--list-models/--dry-run) or
# a board target (image + person video tracking): x5, s100, s100p, s600.
# Each target builds in its own build/<target>/ dir.
#   ./run.sh --target x5 -- --input bus.jpg --save-path result.jpg
#   ./run.sh --target s100 -- --input persons.mp4 --track --records out.tsv
#   ./run.sh --target host -- --list-models
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
  echo "usage: run.sh --target host|x5|s100|s100p|s600 -- [fasterrcnn options]" >&2
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
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DFRCNN_TARGET="$TARGET" \
  -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIR" --target fasterrcnn -j >/dev/null

exec "$BUILD_DIR/fasterrcnn" "$@"
