#!/bin/bash
# Run the sample with a local model or one installed on the board.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly MODEL_DIR="${SCRIPT_DIR}/../../model"
readonly BOARD_DIR="/opt/hobot/model/x5/basic"
readonly MODEL_NAME="whisper_encoder_tiny_80x3000_x5.bin"
readonly LEGACY_NAME="whisper_encoder_tiny_80x3000_nv12.bin"
readonly -a CANDIDATES=(
  "${MODEL_DIR}/${MODEL_NAME}"
  "${BOARD_DIR}/${MODEL_NAME}"
  "${MODEL_DIR}/${LEGACY_NAME}"
  "${BOARD_DIR}/${LEGACY_NAME}"
)

args=()
for path in "${CANDIDATES[@]}"; do
  if [[ -f "${path}" ]]; then
    args+=(--encoder-model-path "${path}")
    break
  fi
done

readonly LEGACY_DECODER="${BOARD_DIR}/whisper_decoder_tiny_80x3000_nv12.bin"
readonly ONNX_DECODER="${MODEL_DIR}/whisper_decoder_tiny.onnx"
if [[ ! -f "${ONNX_DECODER}" && -f "${LEGACY_DECODER}" ]]; then
  args+=(--decoder-model-path "${LEGACY_DECODER}")
fi

exec python3 "${SCRIPT_DIR}/main.py" "${args[@]}" "$@"
