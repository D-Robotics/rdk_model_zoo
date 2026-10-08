# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLO26 rotated-box stages with validated raw outputs and explicit image geometry."""

from dataclasses import dataclass, field
from typing import Optional, List
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import (
    PlatformProfile,
    resolve_platform,
)
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    LTRBOBBContract,
    ModelSelection,
)
from samples.vision.ultralytics_yolo.runtime.python.model_runner import build_runner
from samples.vision.ultralytics_yolo.runtime.python.detect import YoloDetect
from samples.vision.ultralytics_yolo.runtime.python.detection_io import (
    _size_from_runner,
    _normalise_grids,
    _semantic_outputs,
    _transform_for_postprocess,
)
from samples.vision.ultralytics_yolo.runtime.python.obb_decode import decode_obb


@dataclass
class YOLO26OBBConfig:
    """Configuration for the YOLO26 OBB model.

    Attributes:
        model_path: Path to the compiled `.hbm` model.
        score_thres: Confidence threshold for filtering.
        nms_thres: IoU threshold for rotated NMS.
        angle_sign: Multiplier for angle decoding.
        angle_offset: Offset in degrees to add to decoded angle.
        regularize: Whether to regularize boxes (w > h).
        resize_type: Image resize strategy (0=stretch, 1=letterbox).
        strides: Feature map strides.
    """

    model_path: str
    score_thres: float = 0.25
    nms_thres: float = 0.2
    angle_sign: float = 1.0
    angle_offset: float = 0.0
    regularize: bool = True
    resize_type: int = 1
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[tuple] = None
    classes_num: int = 15

    contract: Optional[LTRBOBBContract] = None


class YOLO26OBB:
    """Return independent {rrect, score, id} records; rrect angle is in radians."""

    task = "obb"

    def __init__(self, config: YOLO26OBBConfig, runner=None):
        self.cfg = config
        requested = config.contract or LTRBOBBContract(
            classes=config.classes_num, strides=config.strides
        )
        if runner is None:
            runner = build_runner(
                ModelSelection(
                    config.model_path,
                    target=getattr(config.platform, "key", None),
                    platform=config.platform,
                    task="obb",
                    contract=requested,
                    input_shape=config.input_shape,
                )
            )
        self.runner = runner
        self.binding = getattr(runner, "binding", None)
        if self.binding is None or self.binding.contract.task != "obb":
            raise ValueError("YOLO26OBB requires a bound OBB runner.")
        self.contract = self.binding.contract
        self.profile = self.binding.selection.profile
        if isinstance(self.profile, str):
            self.profile = resolve_platform(self.profile)
        if config.platform is not None:
            profile = (
                resolve_platform(config.platform)
                if isinstance(config.platform, str)
                else config.platform
            )
            if profile.key != self.profile.key:
                raise ValueError("OBB config target conflicts with the runner binding.")
        self.model = runner.model
        self.model_name = runner.model_name
        self.input_adapter = runner.input_adapter
        self.input_h, self.input_w = _size_from_runner(runner, config)
        self.input_size = (self.input_h, self.input_w)
        _normalise_grids(None, self.input_size, self.contract.strides)
        if self.input_h != self.input_w:
            raise ValueError("Published YOLO26 OBB requires square model input.")
        self.input_names = tuple(runner.input_names)
        self.output_names = tuple(runner.output_names)
        self.input_shapes = dict(runner.input_shapes)

    # Borrow the readable DFL stage implementations (this class provides the
    # same runner/binding attributes, and image transport plus the raw model
    # call are genuinely protocol-independent) together with their
    # compatibility aliases; only the OBB post-processing below is
    # protocol-specific.  The full orchestration stays visible in predict.
    preprocess = YoloDetect.preprocess
    infer = YoloDetect.infer
    pre_process = YoloDetect.pre_process
    forward = YoloDetect.forward
    set_scheduling_params = YoloDetect.set_scheduling_params

    def postprocess(
        self,
        outputs,
        ori_w=None,
        ori_h=None,
        score_thres=None,
        nms_thres=None,
        transform=None,
    ):
        """Decode raw radians, apply platform rotated NMS and restore this image."""
        context = _transform_for_postprocess(
            transform, ori_w, ori_h, self.input_size, self.cfg.resize_type
        )
        semantic = _semantic_outputs(self.binding, self.contract, outputs, "YOLO26 OBB")
        return decode_obb(
            semantic,
            self.contract,
            context,
            self.cfg.score_thres if score_thres is None else score_thres,
            self.cfg.nms_thres if nms_thres is None else nms_thres,
            angle_sign=self.cfg.angle_sign,
            angle_offset=self.cfg.angle_offset,
            regularize=self.cfg.regularize,
            platform_family=self.profile.family,
        )

    def predict(self, img, image_format="BGR", score_thres=None, nms_thres=None):
        """Compose the three stages and return owned rotated records.

        preprocess (shared image transport) → infer (one raw model call) →
        postprocess (the OBB-specific rotated decode below).
        """
        prepared = self.preprocess(img, image_format)
        outputs = self.infer(prepared)
        return self.postprocess(
            outputs,
            score_thres=score_thres,
            nms_thres=nms_thres,
            transform=prepared.transform,
        )

    def __call__(self, img, image_format="BGR", score_thres=None, nms_thres=None):
        return self.predict(img, image_format, score_thres, nms_thres)

    def post_process(
        self,
        outputs,
        ori_w=None,
        ori_h=None,
        score_thres=None,
        nms_thres=None,
        transform=None,
    ):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(
            outputs, ori_w, ori_h, score_thres, nms_thres, transform)
