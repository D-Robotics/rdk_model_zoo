# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""File decoding outside KWS inference stages."""

from pathlib import Path
import numpy as np


def load_audio(path):
    import soundfile as sf

    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"Missing audio file: {path}")
    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.shape[1] != 1:
        raise ValueError(
            "Published KWS accepts mono audio; convert channels explicitly"
        )
    return np.array(waveform[:, 0], dtype=np.float32, copy=True), int(rate)
