# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLO26 pose selects direct LTRB/point offsets on the shared pose stages."""

from dataclasses import dataclass, field, replace
from typing import Optional
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import PlatformProfile
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    LTRBPoseContract,
)
from samples.vision.ultralytics_yolo.runtime.python.yolo_pose import YoloPose


@dataclass
class YOLO26PoseConfig:
    """Local artifact, target, confidence/NMS thresholds and resize strategy.

    Library NMS default remains 0.65; CLI and legacy adapters explicitly provide
    their platform defaults. Only strides 8/16/32 and COCO-17 are supported.
    """

    model_path: str
    score_thres: float = 0.25
    nms_thres: float = 0.65
    resize_type: int = 1
    strides: list = field(default_factory=lambda: [8, 16, 32])
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[tuple] = None
    contract: Optional[LTRBPoseContract] = None
    anchor_sizes: Optional[list] = None


class YOLO26Pose(YoloPose):
    """Shared raw stages; five owned arrays with one sigmoid visibility."""

    task = "pose"

    def __init__(self, config: YOLO26PoseConfig, runner=None):
        requested = config.contract or LTRBPoseContract(strides=config.strides)
        effective = getattr(getattr(runner, "binding", None), "contract", requested)
        for contract in (requested, effective):
            if (
                contract.task != "pose"
                or getattr(contract, "box_encoding", None) != "ltrb"
                or contract.nkpt != 17
            ):
                raise ValueError(
                    "YOLO26Pose requires the direct-LTRB COCO-17 pose contract."
                )
        super().__init__(replace(config, contract=requested), runner=runner)
