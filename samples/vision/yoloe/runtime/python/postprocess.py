# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""One postprocessing implementation shared by board inference and float evaluation."""

from utils.py_utils.yoloe26_decode import decode_candidates, restore_masks
from samples.vision.ultralytics_yolo.runtime.python.segmentation_decode import (
    decode_segmentation,
)
from samples.vision.yoloe.runtime.python.decode import decode_x5, validate_semantic
from samples.vision.yoloe.runtime.python.pipeline_io import (
    Result,
    validate_config,
    validate_context,
)


def decode_result(semantic, contract, selection, cfg, context):
    validate_config(selection, cfg)
    validate_context(context, selection, cfg)
    semantic = validate_semantic(semantic, contract)
    if selection.variant.startswith("26"):
        ordered = [semantic[role] for role in contract.required_roles]
        boxes, scores, ids, coefficients = decode_candidates(
            ordered, cfg.score_thres, cfg.max_det, cfg.single_label
        )
        boxes, masks = restore_masks(
            boxes, coefficients, semantic["protos"][0], context
        )
        return Result(boxes, scores, ids, masks, "roi")
    nms = 0.7 if cfg.nms_thres is None else cfg.nms_thres
    if selection.target == "x5":
        values = decode_x5(semantic, contract, context, cfg.score_thres, nms)
        return Result(*values, "full")
    values = decode_segmentation(
        semantic, contract, context, cfg.score_thres, nms, do_morph=cfg.do_morph
    )
    return Result(*values, "roi")
