# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLOE PF stages: preprocessing, one raw call, postprocessing and composition."""

from samples.vision.yoloe.runtime.python.config import Config
from samples.vision.ultralytics_yolo.runtime.python.detection_io import (
    _semantic_outputs,
)
from samples.vision.yoloe.runtime.python.model_binding import runtime_selection
from samples.vision.yoloe.runtime.python.model_runner import build_runner
from samples.vision.yoloe.runtime.python.pipeline_io import (
    Prepared,
    prepare,
    validate_config,
)
from samples.vision.yoloe.runtime.python.postprocess import decode_result


class YOLOE:
    """No downloads, visualization, file access or mutable last-image state."""

    def __init__(self, selection, config=None, *, runner=None):
        self.selection = selection
        self.cfg = config or Config()
        validate_config(selection, self.cfg)
        selected = runtime_selection(selection)
        self.runner = runner if runner is not None else build_runner(selection)
        self.binding = self.runner.binding
        if self.binding.selection != selected or self.runner.input_size != (640, 640):
            raise ValueError("Injected runner does not match the YOLOE selection.")
        self.contract = self.binding.contract

    def preprocess(self, image):
        """BGR uint8 HWC -> NV12 tensors plus immutable per-image context."""
        return prepare(image, self.selection, self.cfg, self.runner)

    def infer(self, prepared):
        """Exactly one runner call; return borrowed raw floating arrays unchanged."""
        return self.runner(
            prepared.tensors if isinstance(prepared, Prepared) else prepared
        )

    def postprocess(self, outputs, context):
        """Raw outputs plus matching context -> owned original-coordinate results."""
        semantic = _semantic_outputs(self.binding, self.contract, outputs, "YOLOE PF")
        return decode_result(semantic, self.contract, self.selection, self.cfg, context)

    def predict(self, image):
        """Compose the public stages, carrying this image's context explicitly."""
        prepared = self.preprocess(image)
        return self.postprocess(self.infer(prepared), prepared.context)

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin
    # aliases of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, image):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image)

    def forward(self, prepared):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(prepared)

    def post_process(self, outputs, context):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs, context)
