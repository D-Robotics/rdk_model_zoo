#!/usr/bin/env bash
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Compatibility entrypoint retained for the publication manifest.  Target
# selection is explicit; this wrapper never guesses or falls back to S100.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/download.sh" "$@"
