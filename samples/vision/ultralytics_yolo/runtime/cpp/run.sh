#!/usr/bin/env bash
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
# Build (out-of-tree) and run the consolidated ultralytics_yolo C++ runtime.
# All arguments are forwarded to the binary; the build target defaults to
# auto-discovered board headers and can be pinned with YOLO_TARGET
# (x5, s100, s100p, s600). Run from the sample root so the default
# test_data/ images resolve.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TARGET=${YOLO_TARGET:-auto}
BUILD_DIR="$SCRIPT_DIR/build/$TARGET"

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DYOLO_TARGET="$TARGET"
cmake --build "$BUILD_DIR" --parallel

exec "$BUILD_DIR/ultralytics_yolo_cpp" "$@"
