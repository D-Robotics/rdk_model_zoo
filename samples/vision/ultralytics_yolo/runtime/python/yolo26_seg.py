# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLO26 selects direct boxes and probability masks on shared segmentation stages."""

from dataclasses import dataclass, field, replace
from typing import Optional
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import PlatformProfile
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    LTRBSegmentationContract,
)
from samples.vision.ultralytics_yolo.runtime.python.yolo_seg import YoloSeg


@dataclass
class YOLO26SegConfig:
    """Local artifact/target and thresholds; 32 coefficients, no mask morphology.

    Library NMS default remains 0.65; CLI supplies its platform default.
    resize_type is 0=stretch or 1=letterbox.
    """

    model_path: str
    classes_num: int = 80
    score_thres: float = 0.25
    nms_thres: float = 0.65
    resize_type: int = 1
    strides: list = field(default_factory=lambda: [8, 16, 32])
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[tuple] = None
    contract: Optional[LTRBSegmentationContract] = None
    anchor_sizes: Optional[list] = None
    do_morph: bool = field(default=False, init=False)


class YOLO26Seg(YoloSeg):
    """Shared raw stages returning owned boxes/scores/IDs and boolean ROI masks."""

    task = "seg"

    def __init__(self, config: YOLO26SegConfig, runner=None):
        requested = config.contract or LTRBSegmentationContract(
            classes=config.classes_num, strides=config.strides
        )
        effective = getattr(getattr(runner, "binding", None), "contract", requested)
        for contract in (requested, effective):
            if (
                contract.task != "segment"
                or getattr(contract, "box_encoding", None) != "ltrb"
                or contract.mces_num != 32
            ):
                raise ValueError(
                    "YOLO26Seg requires the direct-LTRB 32-coefficient segmentation contract."
                )
        super().__init__(replace(config, contract=requested), runner=runner)
