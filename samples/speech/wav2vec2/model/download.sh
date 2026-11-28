#!/bin/bash
# Download the prebuilt wav2vec2 model resources for RDK X5.

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DIR
readonly ARCHIVE="https://archive.d-robotics.cc/downloads/rdk_model_zoo"
readonly BASE_URL="${ARCHIVE}/rdk_x5/wav2vec2"

readonly NAME="wav2vec2_base_1x246000_nv12.bin"
if [[ -f "${DIR}/${NAME}" ]]; then
  echo "[skip] ${NAME} already present"
else
  echo "[download] ${NAME}"
  wget -c -P "${DIR}" "${BASE_URL}/${NAME}"
fi
