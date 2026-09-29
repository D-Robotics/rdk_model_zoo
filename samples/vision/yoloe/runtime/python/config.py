# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Configuration shared by Python stages and native launcher; no image/SDK imports."""

from dataclasses import dataclass
from numbers import Integral
import numpy as np


@dataclass(frozen=True)
class Config:
    score_thres: float = 0.25
    nms_thres: float | None = None
    resize_type: int = 1
    do_morph: bool = False
    max_det: int = 300
    single_label: bool = True


def validate_config(selection, cfg):
    if not np.isfinite(cfg.score_thres) or not 0 < cfg.score_thres < 1:
        raise ValueError("score_thres must be finite and strictly between 0 and 1.")
    if cfg.resize_type not in (0, 1) or not isinstance(cfg.do_morph, bool):
        raise ValueError("resize_type must be 0/1 and do_morph must be boolean.")
    if (
        isinstance(cfg.max_det, bool)
        or not isinstance(cfg.max_det, Integral)
        or not 1 <= cfg.max_det <= 8400
        or not isinstance(cfg.single_label, bool)
    ):
        raise ValueError(
            "max_det must be an integer in 1..8400; single_label must be boolean."
        )
    if selection.variant.startswith("26"):
        if cfg.nms_thres is not None or cfg.resize_type != 1 or cfg.do_morph:
            raise ValueError(
                "YOLOE-26 uses fixed round/114 letterbox, no NMS and no morphology."
            )
    else:
        if cfg.max_det != 300 or not cfg.single_label:
            raise ValueError("max_det and multi-label apply only to YOLOE-26.")
        if cfg.nms_thres is not None and (
            not np.isfinite(cfg.nms_thres) or not 0 <= cfg.nms_thres <= 1
        ):
            raise ValueError("nms_thres must be finite in [0,1].")
        if selection.target == "x5" and cfg.do_morph:
            raise ValueError("Morphology applies only to S YOLOE-11 ROI masks.")
