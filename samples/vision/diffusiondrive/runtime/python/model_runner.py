# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Four-input named transport reusing common identity/hash/SDK ownership."""

from samples._shared.single_array_runner import (
    NamedArrayRunner,
    RuntimeUnavailableError,
)
from samples._shared.platforms import require_execution_target
from samples.vision.diffusiondrive.runtime.python.model_binding import bind_model


class RuntimeModelRunner(NamedArrayRunner):
    def __init__(self, selection, *, runtime=None, runtime_factory=None):
        super().__init__(
            selection,
            binding_loader=bind_model,
            physical_inputs=lambda b: {
                n: (s.shape, s.dtype) for n, s in b.input_transforms.items()
            },
            task_name="DiffusionDrive",
            runtime=runtime,
            runtime_factory=runtime_factory,
            execution_target_gate=require_execution_target,
        )
