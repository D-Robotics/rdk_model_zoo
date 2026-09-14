#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_MODEL="${SAMPLE_ROOT}/model/bayes-e/himloco_go2_bayese_1x270.bin"

use_default_model=true
for argument in "$@"; do
  if [[ "${argument}" == "--model-path" || "${argument}" == --model-path=* ]]; then
    use_default_model=false
    break
  fi
done

if [[ "${use_default_model}" == true && ! -f "${DEFAULT_MODEL}" ]]; then
  bash "${SAMPLE_ROOT}/model/download_model.sh"
fi

cd "${SCRIPT_DIR}"
exec python3 main.py "$@"
