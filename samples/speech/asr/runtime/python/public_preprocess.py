# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pinned XLSR53 frontend with upstream normalization and real padding mask."""
import numpy as np

SAMPLES = 30000
KERNELS = (10, 3, 3, 3, 3, 2, 2)
STRIDES = (5, 2, 2, 2, 2, 2, 2)


def feature_frames(valid_samples):
    """Return the convolution frontend's valid frame count, including zero."""
    if not isinstance(valid_samples, int) or not 0 <= valid_samples <= SAMPLES:
        raise ValueError('Expected 0..30000 valid samples')
    length = valid_samples
    for kernel, stride in zip(KERNELS, STRIDES):
        length = (length - kernel) // stride + 1
    return max(0, length)


def prepare_waveform(waveform):
    """Normalize one 16kHz mono F32 window before padding; retain its mask.

    The caller supplies at most 30000 samples and performs any resampling.
    Public masks use F32 zero/one; the exported model converts them to int64.
    """
    raw = np.asarray(waveform)
    if (raw.dtype != np.float32 or raw.ndim != 1 or not 0 < raw.size <= SAMPLES
            or not np.isfinite(raw).all()):
        raise ValueError('Expected 1..30000 finite mono float32 samples at 16kHz')
    tensor = np.zeros((1, SAMPLES), np.float32)
    mask = np.zeros_like(tensor)
    tensor[0, :raw.size] = ((raw - np.mean(raw))
                            / np.sqrt(np.var(raw) + np.float32(1e-7)))
    mask[0, :raw.size] = 1
    return {'waveform': tensor, 'attention_mask': mask}
