#!/usr/bin/env bash
set -euo pipefail

# Model preparation and dependency setup are explicit. This historical
# launcher only delegates to the canonical-compatible command and never
# changes the host Python environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/main.py" "$@"
