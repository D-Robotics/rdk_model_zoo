#!/bin/bash
# Download the prebuilt whisper model resources for RDK X5.

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DIR
readonly ARCHIVE="https://archive.d-robotics.cc/downloads/rdk_model_zoo"
readonly BASE_URL="${ARCHIVE}/rdk_x5/whisper"

readonly -a FILES=(
  "whisper_encoder_tiny_80x3000_nv12.bin"
  "whisper_decoder_tiny_80x3000_nv12.bin"
  "vocab_en.txt"
  "vocab_zh.txt"
  "mel_80_filters.txt"
)

for name in "${FILES[@]}"; do
  if [[ -f "${DIR}/${name}" ]]; then
    echo "[skip] ${name} already present"
  else
    echo "[download] ${name}"
    wget -c -P "${DIR}" "${BASE_URL}/${name}"
  fi
done
