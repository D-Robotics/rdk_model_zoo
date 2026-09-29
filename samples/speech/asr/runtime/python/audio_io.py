# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Bounded audio-file streaming outside ASR task math."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from samples.speech.asr.runtime.python.frontend import Config, source_chunk_size


@dataclass(frozen=True)
class AudioChunk:
    waveform: np.ndarray
    sample_rate: int
    source_start: int
    index: int


def read_chunks(path, config=Config()):
    import soundfile as sf

    with sf.SoundFile(Path(path).expanduser(), "r") as stream:
        rate = int(stream.samplerate)
        size = source_chunk_size(rate, config)
        if stream.frames <= 0:
            raise ValueError("Audio file contains no frames")
        start = 0
        index = 0
        while True:
            data = stream.read(size, dtype="float32")
            if len(data) == 0:
                break
            yield AudioChunk(data, rate, start, index)
            start += len(data)
            index += 1
