# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Lazy SDK transport; one raw score array from two physical NV12 planes."""
from samples._shared.platforms import require_execution_target
from samples._shared.single_array_runner import SingleArrayRunner
from samples.vision.unetmobilenet.runtime.python.model_binding import bind_model


class RuntimeModelRunner(SingleArrayRunner):
    def __init__(self, selection, *, runtime=None, runtime_factory=None):
        super().__init__(
            selection,
            binding_loader=bind_model,
            physical_inputs=lambda binding: {
                binding.y_name: ((1, 1024, 2048, 1), 'uint8'),
                binding.uv_name: ((1, 512, 1024, 2), 'uint8'),
            },
            task_name='UnetMobileNet',
            runtime=runtime,
            runtime_factory=runtime_factory,
            execution_target_gate=require_execution_target,
        )
