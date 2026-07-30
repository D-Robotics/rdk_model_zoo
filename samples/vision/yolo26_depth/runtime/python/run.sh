#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODEL=${1:-$ROOT/model/bayes-e/yolo26n_depth_bayese_768x768_nv12.bin}
INPUT=${2:-$ROOT/test_data/bus.jpg}
OUTPUT=${3:-$ROOT/test_data/python_result}

python3 "$ROOT/runtime/python/main.py" --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
