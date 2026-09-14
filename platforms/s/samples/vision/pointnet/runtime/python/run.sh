#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}"

SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name 2>/dev/null || echo "s100")
MODEL_SOC="${MODEL_SOC:-s100}"
echo "SOC        : ${SOC}"
echo "Model SOC  : ${MODEL_SOC}"

(cd ../../model && bash download_model.sh "${MODEL_SOC}")

python3 main.py \
  --model-path "../../model/${MODEL_SOC}/pointnet.hbm" \
  --test-pts "../../test_data/chair.pts" \
  --img-save-path "result.png" \
  "$@"
