#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12.yaml}
MODEL=${MODEL:-onnx/pp_liteseg_stdc1_cityscapes_1024x512_sim.onnx}
CAL_SRC=${CAL_SRC:-}

if [[ ! -f "${MODEL}" ]]; then
  echo "Missing ONNX model: ${MODEL}"
  echo "Run onnx_export/export_pp_liteseg_stdc1_onnx.sh first, or place the ONNX file at this path."
  exit 1
fi

if [[ -n "${CAL_SRC}" ]]; then
  python3 prepare_calibration.py \
    --src "${CAL_SRC}" \
    --out calibration_data_rgb_f32_1024x512 \
    --width 1024 \
    --height 512 \
    --num 50
fi

if [[ ! -d calibration_data_rgb_f32_1024x512 ]]; then
  echo "Missing calibration_data_rgb_f32_1024x512."
  echo "Set CAL_SRC=/path/to/images when running this script, or run prepare_calibration.py manually."
  exit 1
fi

hb_mapper checker --model-type onnx --march bayes-e --model "${MODEL}"
hb_mapper makertbin --config "${CONFIG}" --model-type onnx

BIN=ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12_output/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin
if [[ -f "${BIN}" ]]; then
  hb_perf "${BIN}"
  echo "Generated BIN: ${BIN}"
else
  echo "makertbin finished, but expected BIN was not found: ${BIN}"
  find ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12_output -name '*.bin' -print || true
fi
