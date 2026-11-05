# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Prepare the pinned base224 source's bicubic calibration inputs."""
from utils.tools.hf_transformer_conversion import main

if __name__ == '__main__':
    raise SystemExit(main('siglip', 'prepare'))
