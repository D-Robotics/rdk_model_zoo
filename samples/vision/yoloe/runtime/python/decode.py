# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""X5 YOLOE-11 full-image masks; S11 reuses reviewed ROI segmentation math."""

import cv2
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.rdk_yolo_utils import (
    postprocess as post,
)
from samples.vision.ultralytics_yolo.runtime.python.geometry import inverse_boxes


def decode_x5(outputs, contract, context, score_thres, nms_thres):
    """Keep source DFL/NMS, prototype probability crop and full-image bool masks."""
    boxes, scores, ids, coefficients = [], [], [], []
    threshold = -np.log(1.0 / np.clip(score_thres, 1e-6, 1.0 - 1e-6) - 1.0)
    weights = np.arange(16, dtype=np.float32)[None, None, :]
    for stride in (8, 16, 32):
        conf, cls, selected = post.filter_classification(
            outputs[f"cls_{stride}"], threshold
        )
        boxes.append(
            post.decode_boxes(
                outputs[f"box_{stride}"], selected, 640 // stride, stride, weights
            )
        )
        scores.append(conf)
        ids.append(cls)
        coefficients.append(post.filter_mces(outputs[f"mces_{stride}"], selected))
    boxes, scores, ids, coefficients = [
        np.concatenate(x, axis=0) for x in (boxes, scores, ids, coefficients)
    ]
    keep = post.NMS(boxes, scores, ids, nms_thres)
    boxes, scores, ids, coefficients = [
        x[keep] for x in (boxes, scores, ids, coefficients)
    ]
    proto = outputs["protos"][0].transpose(2, 0, 1)
    probabilities = post.sigmoid(
        (coefficients @ proto.reshape(32, -1)).reshape(-1, 160, 160)
    )
    # Source crops probabilities at prototype resolution before either resize.
    bounds = boxes * np.array([0.25, 0.25, 0.25, 0.25])
    x1, y1, x2, y2 = np.split(bounds[:, :, None], 4, axis=1)
    xx = np.arange(160, dtype=np.float32)[None, None, :]
    yy = np.arange(160, dtype=np.float32)[None, :, None]
    probabilities *= (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    h, w = context.original_size
    left, top, _, _ = context.padding
    rh, rw = context.resized_size
    masks = []
    for probability in probabilities:
        padded = cv2.resize(probability, (640, 640), interpolation=cv2.INTER_LINEAR)
        content = padded[top : top + rh, left : left + rw]
        masks.append(cv2.resize(content, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5)
    return (
        inverse_boxes(boxes, context),
        np.array(scores, dtype=np.float32, copy=True),
        np.array(ids, dtype=np.int64, copy=True),
        np.array(masks, dtype=bool).reshape(-1, h, w),
    )


def validate_semantic(outputs, contract):
    """An injected semantic mapping must satisfy the same float contract as the SDK."""
    expected = {
        f"{kind}_{stride}": (1, 640 // stride, 640 // stride, channels)
        for stride in (8, 16, 32)
        for kind, channels in [
            ("cls", 4585),
            ("box", getattr(contract, "box_channels", 64)),
            ("mces", 32),
        ]
    }
    expected["protos"] = (1, 160, 160, 32)
    if set(outputs) != set(expected):
        raise ValueError("YOLOE requires exactly ten semantic output roles.")
    for name, shape in expected.items():
        array = np.asarray(outputs[name])
        if (
            array.shape != shape
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"{name} requires finite float32 {shape}.")
    return outputs
