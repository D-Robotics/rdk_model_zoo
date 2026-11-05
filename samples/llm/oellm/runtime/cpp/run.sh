#!/usr/bin/env bash
set -euo pipefail
sample_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OELLM_RUNTIME_ROOT="${OELLM_RUNTIME_ROOT:-/opt/oellm_runtime_s600}"
if [[ -f "$OELLM_RUNTIME_ROOT/env.sh" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  source "$OELLM_RUNTIME_ROOT/env.sh" 4
fi
exec python3 "$sample_dir/run.py" "$@"
