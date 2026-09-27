# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLOE PF stages: preprocessing, one raw call, postprocessing and composition."""

from dataclasses import dataclass
from samples._shared.yoloe26_decode import decode_candidates, restore_masks
from samples.vision.ultralytics_yolo.runtime.python.detection_io import (
    _semantic_outputs,
)
from samples.vision.ultralytics_yolo.runtime.python.segmentation_decode import (
    decode_segmentation,
)
from samples.vision.yoloe.runtime.python.decode import decode_x5, validate_semantic
from samples.vision.yoloe.runtime.python.model_binding import runtime_selection
from samples.vision.yoloe.runtime.python.model_runner import build_runner
from samples.vision.yoloe.runtime.python.pipeline_io import (
    Prepared,
    Result,
    prepare,
    validate_config,
    validate_context,
)


@dataclass(frozen=True)
class Config:
    score_thres: float = 0.25
    nms_thres: float | None = None
    resize_type: int = 1
    do_morph: bool = False
    max_det: int = 300
    single_label: bool = True


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

    def pre_process(self, image):
        """BGR uint8 HWC -> NV12 tensors plus immutable per-image context."""
        return prepare(image, self.selection, self.cfg, self.runner)

    def forward(self, prepared):
        """Exactly one runner call; return borrowed raw floating arrays unchanged."""
        return self.runner(
            prepared.tensors if isinstance(prepared, Prepared) else prepared
        )

    def post_process(self, outputs, context):
        """Raw outputs plus matching context -> owned original-coordinate results."""
        validate_context(context, self.selection, self.cfg)
        semantic = validate_semantic(
            _semantic_outputs(self.binding, self.contract, outputs, "YOLOE PF"),
            self.contract,
        )
        if self.selection.variant.startswith("26"):
            ordered = [semantic[role] for role in self.contract.required_roles]
            boxes, scores, ids, coefficients = decode_candidates(
                ordered, self.cfg.score_thres, self.cfg.max_det, self.cfg.single_label
            )
            boxes, masks = restore_masks(
                boxes, coefficients, semantic["protos"][0], context
            )
            return Result(boxes, scores, ids, masks, "roi")
        nms = 0.7 if self.cfg.nms_thres is None else self.cfg.nms_thres
        if self.selection.target == "x5":
            values = decode_x5(
                semantic, self.contract, context, self.cfg.score_thres, nms
            )
            return Result(*values, "full")
        values = decode_segmentation(
            semantic,
            self.contract,
            context,
            self.cfg.score_thres,
            nms,
            do_morph=self.cfg.do_morph,
        )
        return Result(*values, "roi")

    def predict(self, image):
        """Compose the public stages, carrying this image's context explicitly."""
        prepared = self.pre_process(image)
        return self.post_process(self.forward(prepared), prepared.context)
