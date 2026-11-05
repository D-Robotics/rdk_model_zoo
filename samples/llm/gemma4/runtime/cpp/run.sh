#!/usr/bin/env bash
set -euo pipefail
sample_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_root="${OELLM_RUNTIME_ROOT:-${OELLM_SDK_ROOT:+${OELLM_SDK_ROOT}/oellm_runtime_s600}}"
if [[ -z "$runtime_root" || ! -f "$runtime_root/env.sh" ]]; then
  echo 'Set OELLM_RUNTIME_ROOT to the supplied SDK 2.0 oellm_runtime_s600 directory.' >&2
  exit 2
fi
runtime_root="$(cd -- "$runtime_root" && pwd)"
# The official environment selects its own libraries and four-core L2M settings.
# SDK 2.0 env.sh reads this variable directly, including under bash nounset.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "$runtime_root/env.sh" 4
cmake -S "$sample_dir" -B "$sample_dir/build" -DOELLM_RUNTIME_ROOT="$runtime_root"
cmake --build "$sample_dir/build" --parallel 4
exec "$sample_dir/build/main" "$@"
