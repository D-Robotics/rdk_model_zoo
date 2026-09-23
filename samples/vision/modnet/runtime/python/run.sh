#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/../../../../../" && pwd)"
cd "$ROOT_DIR"
exec python3 -m samples.vision.modnet.runtime.python.main "$@"
