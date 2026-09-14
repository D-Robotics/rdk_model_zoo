#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BOARD=${BOARD:-s100}
case "$BOARD" in s100|s100p) ;; *) echo 'BOARD must be s100 or s100p' >&2; exit 2;; esac
RUNTIME_ROOT=${OELLM_RUNTIME_ROOT:-${OELLM_SDK_ROOT:+$OELLM_SDK_ROOT/oellm_runtime}}
if [[ -z "$RUNTIME_ROOT" || ! -f "$RUNTIME_ROOT/include/xlm.h" ]]; then
  echo 'Set OELLM_SDK_ROOT to the S100 1.0.0 SDK root or OELLM_RUNTIME_ROOT to its oellm_runtime.' >&2
  exit 1
fi
RUNTIME_ROOT=$(realpath "$RUNTIME_ROOT")
export BOARD MODEL_DIR=${MODEL_DIR:-"$HERE/../../model/$BOARD"}
bash "$HERE/../../model/download_model.sh"
cmake -S "$HERE" -B "$HERE/build" -DOELLM_RUNTIME_ROOT="$RUNTIME_ROOT"
cmake --build "$HERE/build" --parallel 2
export LD_LIBRARY_PATH="$RUNTIME_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec timeout --foreground "${INFERENCE_TIMEOUT:-120}" "$HERE/build/main"   --model-path "$MODEL_DIR/minicpm5-2b_ctx4096_${BOARD}.hbm"   --tokenizer-path "$MODEL_DIR/tokenizer" --template-path "$MODEL_DIR/tokenizer/simple-chat.jinja" "$@"
