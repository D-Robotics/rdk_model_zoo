#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODEL=${1:-$ROOT/model/bayes-e/yolo26n_depth_bayese_768x768_nv12.bin}
INPUT=${2:-$ROOT/test_data/bus.jpg}
OUTPUT=${3:-$ROOT/test_data/cpp_result}
BUILD_DIR=${BUILD_DIR:-$ROOT/runtime/cpp/build}

cmake -S "$ROOT/runtime/cpp" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"$(nproc)"
"$BUILD_DIR/yolo26_depth" "$MODEL" "$INPUT" "$OUTPUT"
