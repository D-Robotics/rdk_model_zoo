# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""File-to-tensor adapters for the source-provided LPRNet fixture format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from samples.vision.lprnet.runtime.python.model_binding import INPUT_SHAPE


def read_float32_input(path: str | Path) -> tuple[np.ndarray, Path]:
    """Read exactly one ``(1,3,24,94)`` float32 tensor and its source path."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    data = np.fromfile(source, dtype=np.float32)
    expected = int(np.prod(INPUT_SHAPE))
    if data.size != expected:
        raise ValueError(f"Expected {expected} float32 values, got {data.size}.")
    return data.reshape(INPUT_SHAPE).copy(), source


__all__ = ["read_float32_input"]
