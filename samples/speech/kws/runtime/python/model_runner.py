# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""KWS binding over the shared raw single-array transport."""

from samples._shared.single_array_runner import SingleArrayRunner
from samples._shared.platforms import require_execution_target
from samples.speech.kws.runtime.python.model_binding import bind_model


class RuntimeModelRunner(SingleArrayRunner):
    def __init__(self, selection, *, runtime_factory=None, runtime=None):
        super().__init__(
            selection,
            binding_loader=bind_model,
            physical_input=lambda binding: ((1, 373, 80), "float32"),
            task_name="KWS",
            runtime_factory=runtime_factory,
            runtime=runtime,
            execution_target_gate=require_execution_target,
        )
