# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Validate observed raw outputs without assuming order or auxiliary semantics."""

from collections.abc import Mapping
import numpy as np


def validate_raw(outputs, binding):
    meta = binding.metadata
    if not isinstance(outputs, Mapping) or set(outputs) != set(meta.output_names):
        raise ValueError("Output names differ from bound metadata")
    for name in meta.output_names:
        value = outputs[name]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != meta.output_shapes[name]
            or value.dtype != np.dtype(meta.output_dtypes[name])
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Invalid raw output shape/type/values: {name}")
