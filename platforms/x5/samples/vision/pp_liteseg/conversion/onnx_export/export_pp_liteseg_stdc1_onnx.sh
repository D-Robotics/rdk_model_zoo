#!/usr/bin/env bash
set -euo pipefail

# Export PP-LiteSeg-STDC1 from PaddleSeg to ONNX with static 1024x512 input.
# Run this script in a Python environment with PaddlePaddle, PaddleSeg, and paddle2onnx installed.
# Set PADDLESEG_DIR to the path of your local PaddleSeg clone before running.

PADDLESEG_DIR=${PADDLESEG_DIR:-PaddleSeg}
CONFIG=${CONFIG:-configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml}
CHECKPOINT=${CHECKPOINT:-}
EXPORT_DIR=${EXPORT_DIR:-../inference_model/pp_liteseg_stdc1_cityscapes_1024x512}
ONNX_DIR=${ONNX_DIR:-../onnx}
ONNX_NAME=${ONNX_NAME:-pp_liteseg_stdc1_cityscapes_1024x512.onnx}

if [[ ! -d "${PADDLESEG_DIR}" ]]; then
  echo "ERROR: PaddleSeg directory not found: ${PADDLESEG_DIR}"
  echo "Clone it first: git clone --depth=1 https://gitee.com/paddlepaddle/PaddleSeg.git"
  exit 1
fi

cd "${PADDLESEG_DIR}"

python -m pip install -r requirements.txt
python -m pip install paddle2onnx onnx onnxsim

EXPORT_ARGS=(
  tools/export.py
  --config "${CONFIG}"
  --save_dir "${EXPORT_DIR}"
)

if [[ -n "${CHECKPOINT}" ]]; then
  EXPORT_ARGS+=(--model_path "${CHECKPOINT}")
fi

python "${EXPORT_ARGS[@]}"

mkdir -p "${ONNX_DIR}"

paddle2onnx \
  --model_dir "${EXPORT_DIR}" \
  --model_filename model.json \
  --params_filename model.pdiparams \
  --save_file "${ONNX_DIR}/${ONNX_NAME}" \
  --opset_version 11 \
  --enable_onnx_checker True

python -m onnxsim \
  "${ONNX_DIR}/${ONNX_NAME}" \
  "${ONNX_DIR}/${ONNX_NAME%.onnx}_sim.onnx" \
  --overwrite-input-shape 1,3,512,1024

echo "ONNX exported to: ${ONNX_DIR}/${ONNX_NAME%.onnx}_sim.onnx"
