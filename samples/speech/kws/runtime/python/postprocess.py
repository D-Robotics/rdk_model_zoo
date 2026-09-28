# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Convert physical output to existing model probabilities and reduce by max."""

import numpy as np
from samples._shared.quantization import apply_output_transform


def keyword_score(raw, binding):
    meta = binding.metadata
    name = binding.output_name
    if (
        not isinstance(raw, np.ndarray)
        or raw.shape != meta.output_shapes[name]
        or raw.dtype != np.dtype(meta.output_dtypes[name])
        or not np.isfinite(raw).all()
    ):
        raise ValueError("KWS raw output differs from bound shape/dtype/finite values")
    mode = "raw_f32" if raw.dtype == np.float32 else "dequant"
    values = apply_output_transform(mode, {name: raw}, meta.output_quants)[name]
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError(
            "KWS expects already activated probabilities in [0,1], not logits"
        )
    return float(np.max(values))
