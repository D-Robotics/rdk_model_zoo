# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export the complete base224 vision model as one FP32 two-output graph."""
from utils.tools.hf_transformer_conversion import main

if __name__ == '__main__':
    raise SystemExit(main('siglip', 'export'))
