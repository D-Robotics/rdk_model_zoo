# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Prepare the pinned Mandarin XLSR53 waveform and attention-mask inputs."""
from utils.tools.hf_transformer_conversion import main

if __name__ == '__main__':
    raise SystemExit(main('asr', 'prepare'))
