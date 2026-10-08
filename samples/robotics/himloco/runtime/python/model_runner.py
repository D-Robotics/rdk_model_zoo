# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Lazy raw transport using shared board/hash gates and scheduling support."""

from utils.py_utils.single_array_runner import NamedArrayRunner
from samples.robotics.himloco.runtime.python.model_binding import (
    bind_model,
    validate_selection,
)


class RuntimeModelRunner(NamedArrayRunner):
    def __init__(self, selection, *, runtime_factory=None):
        validate_selection(selection)
        super().__init__(
            selection,
            binding_loader=bind_model,
            physical_input=lambda binding: ((1, 270), "float32"),
            task_name="HIMLoco",
            runtime_factory=runtime_factory,
        )
