# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pure per-chunk mono, Fourier resampling, z-score and padding operations."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    audio_maxlen: int = 30000
    new_rate: int = 16000


@dataclass(frozen=True)
class PreparedChunk:
    tensor: np.ndarray
    valid_samples: int
    source_samples: int
    source_rate: int


def validate_config(config):
    if (
        type(config.audio_maxlen) is not int
        or type(config.new_rate) is not int
        or (config.audio_maxlen, config.new_rate) != (30000, 16000)
    ):
        raise ValueError("Published ASR uses 30000 target samples at 16000 Hz")


def source_chunk_size(sample_rate, config=Config()):
    validate_config(config)
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    return (config.audio_maxlen * sample_rate + config.new_rate - 1) // config.new_rate


def prepare_chunk(waveform, sample_rate, config=Config()):
    limit = source_chunk_size(sample_rate, config)
    if (
        not isinstance(waveform, np.ndarray)
        or waveform.ndim not in (1, 2)
        or waveform.dtype != np.float32
        or not 0 < len(waveform) <= limit
        or (waveform.ndim == 2 and waveform.shape[1] < 1)
        or not np.isfinite(waveform).all()
    ):
        raise ValueError(
            "Expected finite float32 [frames] or [frames,channels] within one source chunk"
        )
    mono = waveform.mean(axis=1) if waveform.ndim == 2 else waveform.copy()
    if sample_rate != config.new_rate:
        from scipy.signal import resample

        count = round(len(mono) * config.new_rate / sample_rate)
        if count < 1:
            raise ValueError("Chunk duration is too short to form one target sample")
        mono = resample(mono, count)
    normalized = (mono - np.mean(mono)) / np.sqrt(np.var(mono) + 1e-5)
    if not np.isfinite(normalized).all():
        raise ValueError("Normalization produced nonfinite values")
    count = min(len(normalized), config.audio_maxlen)
    tensor = np.zeros((1, config.audio_maxlen), np.float32)
    tensor[0, :count] = normalized[:count]
    return PreparedChunk(tensor, count, len(waveform), sample_rate)
