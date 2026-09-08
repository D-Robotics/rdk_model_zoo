#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_ROOT=${OELLM_RUNTIME_ROOT:-${OELLM_SDK_ROOT:+$OELLM_SDK_ROOT/oellm_runtime}}
if [[ -z "$RUNTIME_ROOT" || ! -f "$RUNTIME_ROOT/include/oellm_runtime_basic/oellm_runtime.h" ]]; then
  echo "ERROR: set OELLM_SDK_ROOT to the extracted SDK root, or OELLM_RUNTIME_ROOT to oellm_runtime." >&2
  exit 1
fi
RUNTIME_ROOT=$(realpath "$RUNTIME_ROOT")
MODEL_DIR=${MODEL_DIR:-"$HERE/../../model/s600"}
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    --model_path=*) MODEL_DIR=${ARGS[i]#*=} ;;
    --model_path) MODEL_DIR=${ARGS[i+1]:?--model_path requires a directory} ;;
  esac
done
MODEL_DIR=$(realpath -m "$MODEL_DIR")
export MODEL_DIR
bash "$HERE/../../model/download_model.sh"
cmake -S "$HERE" -B "$HERE/build" -DOELLM_RUNTIME_ROOT="$RUNTIME_ROOT"
cmake --build "$HERE/build" --parallel 4
export LD_LIBRARY_PATH="$RUNTIME_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6
exec "$HERE/build/main" --model_path="$MODEL_DIR" "$@"
