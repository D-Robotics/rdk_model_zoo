# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""ASR binding over the shared raw single-array transport."""

from utils.py_utils.single_array_runner import SingleArrayRunner
from utils.py_utils.platforms import require_execution_target
from samples.speech.asr.runtime.python.model_binding import bind_model


class RuntimeModelRunner(SingleArrayRunner):
    def __init__(self, selection, *, runtime_factory=None, runtime=None):
        super().__init__(
            selection,
            binding_loader=bind_model,
            physical_input=lambda binding: ((1, 30000), "float32"),
            task_name="ASR",
            runtime_factory=runtime_factory,
            runtime=runtime,
            execution_target_gate=require_execution_target,
        )
