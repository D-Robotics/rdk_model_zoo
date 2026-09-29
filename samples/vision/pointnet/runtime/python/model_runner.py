# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""PointNet physical input declaration over the shared lazy array transport."""
from samples._shared.platforms import require_execution_target
from samples._shared.single_array_runner import SingleArrayRunner, RuntimeUnavailableError
from samples.vision.pointnet.runtime.python.model_binding import bind_model


class RuntimeModelRunner(SingleArrayRunner):
    """Preserve the sample runner API and leave semantic binding sample-local."""

    def __init__(self, selection, *, runtime_factory=None, runtime=None):
        super().__init__(
            selection,
            binding_loader=bind_model,
            physical_input=lambda binding: (binding.metadata.input_shapes[binding.input_name], "float32"),
            task_name="PointNet",
            runtime_factory=runtime_factory,
            runtime=runtime,
            execution_target_gate=require_execution_target,
        )


__all__ = ["RuntimeModelRunner", "RuntimeUnavailableError"]
