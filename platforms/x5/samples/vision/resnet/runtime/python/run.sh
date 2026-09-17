#!/usr/bin/env bash
set -euo pipefail

# Model preparation is explicit. This historical launcher only delegates to
# the compatibility command, which supplies the old X5 defaults.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/main.py" "$@"
