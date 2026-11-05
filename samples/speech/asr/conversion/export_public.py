# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export all Mandarin CTC logits with the complete dynamic padding mask."""
from utils.tools.hf_transformer_conversion import main

if __name__ == '__main__':
    raise SystemExit(main('asr', 'export'))
