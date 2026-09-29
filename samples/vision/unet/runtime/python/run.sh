#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../../../../.." && pwd)"
cd "$ROOT_DIR"
exec python3 -m samples.vision.unet.runtime.python.main "$@"
