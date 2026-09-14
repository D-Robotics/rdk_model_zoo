#!/bin/bash
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_MODEL="${SAMPLE_ROOT}/model/unet_resnet18_voc_512x512_nv12.bin"

use_default_model=true
for argument in "$@"; do
  if [[ "${argument}" == "--model-path" || "${argument}" == --model-path=* ]]; then
    use_default_model=false
    break
  fi
done

if [[ "${use_default_model}" == true && ! -f "${DEFAULT_MODEL}" ]]; then
  bash "${SAMPLE_ROOT}/model/download_model.sh" resnet18
fi

cd "${SCRIPT_DIR}"
exec python3 main.py "$@"
