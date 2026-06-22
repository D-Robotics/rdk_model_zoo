#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}"

(cd ../../model && bash download_model.sh s100)

python3 main.py \
  --model-path "../../model/s100/r3d_18.hbm" \
  --test-clip "../../test_data/video0.npy" \
  --label-file "../../test_data/kinetics_classnames.json" \
  --top-k 5 \
  "$@"
