# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Reuse the Ultralytics raw runner, with exact YOLOE asset and float32 gates."""

from collections.abc import Mapping
import numpy as np
from utils.py_utils.assets import verify_asset_file, sha256_file
from samples.vision.ultralytics_yolo.runtime.python.model_runner import (
    ModelRunner,
    build_runner as common_runner,
)
from samples.vision.yoloe.runtime.python.model_binding import runtime_selection


def build_runner(selection, *, runtime_loader=None):
    """An explicit injected SDK loader is the host-test seam, never board evidence."""
    selected = runtime_selection(selection)
    if runtime_loader is None:
        if not selection.published_float and not selection.local_float:
            raise ValueError(
                "Published S YOLOE outputs are quantized; prepare and identify a local float model. See conversion/README.md."
            )
        from utils.py_utils.platforms import require_execution_target

        require_execution_target(selection.target)
        if selection.local_float:
            if sha256_file(selection.model_path) != selection.local_float_sha256:
                raise ValueError("Local float SHA-256 mismatch.")
        else:
            verify_asset_file(selection.asset, selection.model_path)
    runner = common_runner(selected, runtime_loader)
    if runner.input_size != (640, 640):
        raise ValueError("YOLOE requires a 640-square input.")
    shapes = []
    for stride in (8, 16, 32):
        shapes.extend(
            (1, 640 // stride, 640 // stride, c)
            for c in (4585, 4 if selection.variant.startswith("26") else 64, 32)
        )
    shapes.append((1, 160, 160, 32))
    actual = [runner.metadata.output_shapes[n] for n in runner.metadata.output_names]
    if sorted(actual) != sorted(shapes) or any(
        runner.metadata.output_dtypes[n] != np.dtype("float32")
        for n in runner.metadata.output_names
    ):
        raise ValueError(
            "YOLOE requires the ten declared NHWC float32 outputs, including NHWC prototype."
        )
    return YOLOERunner(runner.model, runner.binding, runner.metadata)


class YOLOERunner(ModelRunner):
    """Validate physical NV12 input before reusing the common raw SDK call."""

    def __call__(self, tensors):
        if not isinstance(tensors, Mapping) or set(tensors) != {self.model_name}:
            raise ValueError("Expected exactly the bound model input mapping.")
        inputs = tensors[self.model_name]
        if not isinstance(inputs, Mapping) or set(inputs) != set(self.input_names):
            raise ValueError("NV12 input names differ from the binding.")
        for name in self.input_names:
            array = np.asarray(inputs[name])
            shape = (614400,) if self.input_adapter.packed else self.input_shapes[name]
            if (
                array.shape != shape
                or array.dtype != np.uint8
                or not array.flags.c_contiguous
            ):
                raise ValueError(f"{name} requires contiguous uint8 NV12 {shape}.")
        return super().__call__(tensors)
