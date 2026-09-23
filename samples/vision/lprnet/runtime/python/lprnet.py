# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""LPRNet task stages and source CTC-style plate decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from samples.vision.lprnet.runtime.python.model_binding import INPUT_SHAPE, ModelBinding
from samples.vision.lprnet.runtime.python.tensor_io import read_float32_input


CHARS = (
    "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
    "0123456789ABCDEFGHJKLMNPQRSTUV WXYZIO-"
).replace(" ", "")
BLANK_INDEX = len(CHARS) - 1


@dataclass(frozen=True)
class PreparedInput:
    """One owned tensor mapping and its per-call input context."""

    tensors: Mapping[str, np.ndarray]
    context: Path


def decode_plate(logits: np.ndarray) -> str:
    """Apply source argmax, consecutive deduplication, and blank removal."""

    value = np.asarray(logits)
    if value.shape != (68, 18):
        raise ValueError(f"Expected squeezed logits (68, 18), got {value.shape}.")
    labels = np.argmax(value, axis=0)
    decoded: list[int] = []
    previous = int(labels[0])
    if previous != BLANK_INDEX:
        decoded.append(previous)
    for current_value in labels:
        current = int(current_value)
        if current == previous or current == BLANK_INDEX:
            if current == BLANK_INDEX:
                previous = current
            continue
        decoded.append(current)
        previous = current
    return "".join(CHARS[index] for index in decoded)


class LPRNetTask:
    """Four-stage LPRNet task: binary input, raw logits, and CTC plate text."""

    def __init__(self, runner, binding: ModelBinding):
        self.runner = runner
        self.binding = binding

    def pre_process(self, test_bin: str | Path) -> PreparedInput:
        """Read the source-provided float32 binary input without image transforms."""

        tensor, path = read_float32_input(test_bin)
        return PreparedInput({self.binding.input_name: tensor}, path)

    def forward(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Run the selected model and return an owned raw float32 logits array."""

        return self.runner(tensors)

    def post_process(self, raw: np.ndarray) -> str:
        """Decode raw ``(1,68,18)`` logits with source CTC semantics."""

        value = np.asarray(raw)
        if value.shape != (1, 68, 18) or value.dtype != np.float32:
            raise ValueError(f"Expected raw float32 logits (1, 68, 18), got {value.shape}/{value.dtype}.")
        return decode_plate(value[0])

    def predict(self, test_bin: str | Path) -> str:
        """Run pre_process, forward, and post_process for one binary input."""

        prepared = self.pre_process(test_bin)
        return self.post_process(self.forward(prepared.tensors))


LPRNet = LPRNetTask

__all__ = ["BLANK_INDEX", "CHARS", "LPRNet", "LPRNetTask", "PreparedInput", "decode_plate"]
