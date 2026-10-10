#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../../../.." && pwd)"
cd "$ROOT_DIR"
exec "${PYTHON:-python3}" -m samples.vision.diffusiondrive.evaluator.run_cases "$@"
