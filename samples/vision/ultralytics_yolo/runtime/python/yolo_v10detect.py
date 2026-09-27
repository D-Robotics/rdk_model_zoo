# Copyright (c) 2025 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""S-series YOLOv10: the shared DFL detector with an enforced NMS-free contract.

The X5 family dispatcher deliberately keeps its historical NMS detector. This
class is the S protocol adapter, not a second implementation of the stages.
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    DFLDetectionContract,
)
from samples.vision.ultralytics_yolo.runtime.python.yolo_detect import YoloDetect
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import PlatformProfile


@dataclass
class YoloV10DetectConfig:
    """Published S v10 DFL logits: 16 bins, strides 8/16/32 and no NMS.

    Geometry comes from metadata. resize_type is 0=stretch or 1=letterbox;
    classes_num defaults to COCO-80 and score_thres to 0.25.
    """

    model_path: str
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[Tuple[int, int]] = None
    classes_num: int = 80
    resize_type: int = 1
    score_thres: float = 0.25
    reg: int = 16
    strides: list = field(default_factory=lambda: [8, 16, 32])
    anchor_sizes: Optional[list] = None
    contract: Optional[DFLDetectionContract] = None
    nms_thres: Optional[float] = field(default=None, init=False)


class YoloV10Detect(YoloDetect):
    """Reuse all three DFL stages, preserving every score-qualified anchor.

    Results follow stride/grid traversal rather than score sorting or Top-K.
    PreparedDetection carries per-image geometry; results own their arrays.
    """

    def __init__(self, config: YoloV10DetectConfig, runner=None):
        requested = config.contract or DFLDetectionContract(
            classes=config.classes_num,
            reg_bins=config.reg,
            strides=config.strides,
            nms="none",
        )
        effective = getattr(getattr(runner, "binding", None), "contract", requested)
        for contract in (requested, effective):
            if (
                contract.task != "detect"
                or contract.nms != "none"
                or getattr(contract, "reg_bins", None) != 16
                or tuple(contract.strides) != (8, 16, 32)
                or contract.classification != "logits"
                or contract.box_distribution != "logits"
            ):
                raise ValueError(
                    "S YOLOv10 requires DFL logits, 16 bins, strides 8/16/32 and no NMS."
                )
        super().__init__(replace(config, contract=requested), runner=runner)
        if self.input_h != self.input_w:
            raise ValueError("Published S YOLOv10 requires square model input.")
