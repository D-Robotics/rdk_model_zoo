#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIZE=${1:-n}
IMAGE=${2:-"$SCRIPT_DIR/../../test_data/office_desk.jpg"}
OUTPUT=${3:-result.jpg}
MARCH=$(bash "$SCRIPT_DIR/../../model/download_model.sh" --detect)
bash "$SCRIPT_DIR/../../model/download_model.sh" "$MARCH" "$SIZE"
MODEL_DIR="$SCRIPT_DIR/../../model/$MARCH"
cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$SCRIPT_DIR/build" -j2
exec "$SCRIPT_DIR/build/yoloe26seg" \
  "$MODEL_DIR/yoloe_26${SIZE}_seg_pf_${MARCH//-/}_640x640_nv12.hbm" \
  "$MODEL_DIR/yoloe_26${SIZE}_seg_pf.names" "$IMAGE" "$OUTPUT"
