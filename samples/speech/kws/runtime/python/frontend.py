# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Fixed MDTC feature protocol, independent of file I/O and board SDK."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    audio_maxlen: int = 60000
    frame_shift: int = 10
    frame_length: int = 25
    n_mels: int = 80


def validate_config(config):
    values = (
        config.audio_maxlen,
        config.frame_shift,
        config.frame_length,
        config.n_mels,
    )
    if any(type(v) is not int for v in values) or values != (60000, 10, 25, 80):
        raise ValueError(
            "Published KWS requires audio-maxlen=60000, frame-shift=10, frame-length=25, n-mels=80"
        )


def prepare_waveform(waveform, sample_rate, config):
    validate_config(config)
    if type(sample_rate) is not int or sample_rate != 16000:
        raise ValueError("KWS requires 16000 Hz audio; resampling is not implicit")
    if (
        not isinstance(waveform, np.ndarray)
        or waveform.ndim != 1
        or waveform.size == 0
        or waveform.dtype != np.float32
        or not np.isfinite(waveform).all()
    ):
        raise ValueError("Expected nonempty finite mono float32 waveform [N]")
    if np.any(np.abs(waveform) > 1):
        raise ValueError("Expected normalized PCM amplitudes in [-1,1]")
    output = np.zeros((1, config.audio_maxlen), np.float32)
    count = min(waveform.size, config.audio_maxlen)
    output[0, :count] = waveform[:count]
    return output


def paddle_fbank(waveform, config):
    """Use the source PaddleAudio frontend lazily, with explicit source defaults."""
    import paddle
    from paddleaudio.compliance.kaldi import fbank

    return fbank(
        waveform=paddle.to_tensor(waveform),
        sr=16000,
        frame_shift=config.frame_shift,
        frame_length=config.frame_length,
        n_mels=config.n_mels,
    ).numpy()


def prepare_features(waveform, sample_rate, config, binding, frontend=None):
    samples = prepare_waveform(waveform, sample_rate, config)
    features = (frontend or paddle_fbank)(samples, config)
    if (
        not isinstance(features, np.ndarray)
        or features.shape != (373, 80)
        or features.dtype != np.float32
        or not np.isfinite(features).all()
    ):
        raise ValueError("Frontend must produce finite float32 [373,80] features")
    tensor = np.array(features[None], dtype=np.float32, order="C", copy=True)
    if tensor.shape != binding.metadata.input_shapes[binding.input_name]:
        raise ValueError("Extracted feature shape differs from model metadata")
    return {binding.input_name: tensor}
