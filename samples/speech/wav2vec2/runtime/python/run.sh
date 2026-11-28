#!/bin/bash
# Run the sample with a local model or one installed on the board.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly MODEL_DIR="${SCRIPT_DIR}/../../model"
readonly BOARD_DIR="/opt/hobot/model/x5/basic"
readonly MODEL_NAME="wav2vec2_base_960h_1x65535_x5.bin"
readonly LEGACY_NAME="wav2vec2_base_1x246000_nv12.bin"
readonly -a CANDIDATES=(
  "${MODEL_DIR}/${MODEL_NAME}"
  "${BOARD_DIR}/${MODEL_NAME}"
  "${MODEL_DIR}/${LEGACY_NAME}"
  "${BOARD_DIR}/${LEGACY_NAME}"
)

args=()
for path in "${CANDIDATES[@]}"; do
  if [[ -f "${path}" ]]; then
    args+=(--model-path "${path}")
    break
  fi
done

exec python3 "${SCRIPT_DIR}/main.py" "${args[@]}" "$@"
