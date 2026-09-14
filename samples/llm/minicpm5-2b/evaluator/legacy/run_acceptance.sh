#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BOARD=${BOARD:-s100}
case "$BOARD" in s100|s100p) ;; *) echo 'BOARD must be s100 or s100p' >&2; exit 2;; esac
RUNTIME_ROOT=${OELLM_RUNTIME_ROOT:-${OELLM_SDK_ROOT:+$OELLM_SDK_ROOT/oellm_runtime}}
: "${RUNTIME_ROOT:?Set OELLM_RUNTIME_ROOT or OELLM_SDK_ROOT to SDK 1.0.0}"
export BOARD MODEL_DIR=${MODEL_DIR:-"$HERE/../../model/$BOARD"}
bash "$HERE/../../model/download_model.sh"
mkdir -p "$HERE/build"
g++ -std=c++17 -Wall -Wextra -Werror -I"$RUNTIME_ROOT/include" \
  "$HERE/acceptance.cc" -L"$RUNTIME_ROOT/lib" -lxlm -o "$HERE/build/acceptance"
export LD_LIBRARY_PATH="$RUNTIME_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec timeout --foreground "${INFERENCE_TIMEOUT:-900}" "$HERE/build/acceptance" \
  "$MODEL_DIR/minicpm5-2b_ctx4096_${BOARD}.hbm" "$MODEL_DIR/tokenizer" \
  "$HERE/../../test_data/generation-reference.json" "$HERE/../../test_data/legacy-long-prompts.json"
