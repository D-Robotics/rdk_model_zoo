#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}"

(cd ../../model && bash download_model.sh s100)

python3 main.py \
  --model-path "../../model/s100/resnet152_224x224_nv12.hbm" \
  --test-img "../../test_data/zebra_cls.jpg" \
  --label-file "../../../../../datasets/imagenet/imagenet_classes.names" \
  --top-k 5 \
  "$@"
