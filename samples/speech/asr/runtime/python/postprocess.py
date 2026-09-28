# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Physical output validation/conversion followed by explicitly selected decoding."""

import numpy as np
from samples._shared.quantization import apply_output_transform
from samples.speech.asr.runtime.python.decoding import decode_logits


def transcribe(raw, binding, vocabulary, mode):
    name = binding.output_name
    meta = binding.metadata
    if (
        not isinstance(raw, np.ndarray)
        or raw.shape != meta.output_shapes[name]
        or raw.dtype != np.dtype(meta.output_dtypes[name])
        or not np.isfinite(raw).all()
    ):
        raise ValueError("ASR raw logits differ from bound shape/dtype/values")
    transform = "raw_f32" if raw.dtype == np.float32 else "dequant"
    logits = apply_output_transform(transform, {name: raw}, meta.output_quants)[name]
    return decode_logits(np.asarray(logits, dtype=np.float32), vocabulary, mode)
