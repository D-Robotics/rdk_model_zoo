# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Audio decoding and fixed-length waveform preparation for the ASR samples."""

from __future__ import annotations

import os
from typing import BinaryIO

import numpy as np


def read_audio(source: str | os.PathLike | BinaryIO) -> tuple[np.ndarray, int]:
    """Decode an audio file and mix its channels to mono.

    Args:
        source: Path or open binary stream containing encoded audio.

    Returns:
        A one-dimensional float32 waveform and its sample rate in hertz.
    """
    try:
        import soundfile
    except ImportError:
        # The X5 toolchain can read PCM and float WAV files with SciPy alone.
        from scipy.io import wavfile

        sample_rate, pcm = wavfile.read(source)
        audio = pcm.astype(np.float32)
        if pcm.dtype.kind in "iu":
            midpoint = float(2**(pcm.dtype.itemsize * 8 - 1))
            if pcm.dtype.kind == "u":
                audio -= midpoint
            audio /= midpoint
    else:
        audio, sample_rate = soundfile.read(source, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sample_rate)


def fit_waveform(audio: np.ndarray, num_samples: int) -> np.ndarray:
    """Trim or zero-pad a mono waveform to the model's input length.

    Args:
        audio: Mono waveform in sample order.
        num_samples: Number of samples required by the model.

    Returns:
        A one-dimensional float32 array with exactly `num_samples` elements.

    Raises:
        ValueError: If the requested sample count is not positive.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)[:num_samples]
    if audio.size < num_samples:
        audio = np.pad(audio, (0, num_samples - audio.size))
    return audio


def load_waveform(source: str | os.PathLike | BinaryIO,
                  target_sr: int = 16000,
                  num_samples: int | None = None) -> np.ndarray:
    """Decode and resample mono audio, optionally fitting a fixed length.

    Args:
        source: Audio file path or binary stream.
        target_sr: Output sample rate in hertz.
        num_samples: Output sample count, or None to keep the resampled length.

    Returns:
        A one-dimensional float32 waveform at the requested rate.
    """
    audio, sample_rate = read_audio(source)
    if sample_rate != target_sr:
        from scipy import signal

        length = round(audio.size * target_sr / sample_rate)
        audio = signal.resample(audio, length).astype(np.float32)
    if num_samples is not None:
        audio = fit_waveform(audio, num_samples)
    return audio


def wav2vec2_input(audio: np.ndarray, num_samples: int) -> np.ndarray:
    """Normalize a fixed-length waveform for the Base 960h feature extractor.

    Args:
        audio: Mono float waveform at 16 kHz.
        num_samples: Input length, including any zero padding.

    Returns:
        A float32 waveform standardized using all samples, including padding.
    """
    waveform = fit_waveform(audio, num_samples)
    scale = np.sqrt(waveform.var() + np.float32(1e-7))
    return (waveform - waveform.mean()) / scale
