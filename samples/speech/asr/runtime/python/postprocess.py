# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Physical output validation/conversion followed by explicitly selected decoding."""

import numpy as np
from samples._shared.quantization import apply_output_transform, dequantize_tensor
from samples.speech.asr.runtime.python.decoding import (
    decode_exact_logits,
    decode_logits,
)


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
    if raw.dtype == np.float32:
        logits = apply_output_transform("raw_f32", {name: raw}, meta.output_quants)[name]
        return decode_logits(np.asarray(logits, dtype=np.float32), vocabulary, mode)
    quant = meta.output_quants.get(name)
    if quant is None:
        raise ValueError(
            f"Integer ASR output {name!r} has no quantization descriptor in "
            "output_quants; F32 artifacts use the raw_f32 transform."
        )
    # Integer affine scores must keep their ordering through argmax: float32
    # rounds adjacent int32 magnitudes (e.g. 2**24 vs 2**24 + 1) into artificial
    # ties, so dequantization runs at float64 and the exact decoder takes over.
    return decode_exact_logits(
        dequantize_tensor(raw, quant, dtype="float64"), vocabulary, mode
    )
