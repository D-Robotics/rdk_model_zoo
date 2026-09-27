# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pure rotated-box decoding, source NMS policies and explicit inverse geometry."""

import math
import cv2
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.decode import sigmoid


def _cv_rectangle(box):
    return (
        (float(box[0]), float(box[1])),
        (float(box[2]), float(box[3])),
        float(math.degrees(box[4])),
    )


def _rotated_iou(left, right):
    status, points = cv2.rotatedRectangleIntersection(
        _cv_rectangle(left), _cv_rectangle(right)
    )
    if status <= 0 or points is None:
        return 0.0
    intersection = cv2.contourArea(points)
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return 0.0 if union <= 0 else intersection / union


def _rotated_nms(boxes, scores, ids, score_thres, nms_thres, family):
    if family != "x5":
        kept = cv2.dnn.NMSBoxesRotated(
            [_cv_rectangle(b) for b in boxes], scores, score_thres, nms_thres
        )
        return np.asarray(kept, dtype=np.int64).reshape(-1).tolist()
    keep = []
    score_array = np.asarray(scores, np.float32)
    ids = np.asarray(ids, np.int32)
    for cid in np.unique(ids):
        indices = np.where(ids == cid)[0]
        order = indices[np.argsort(score_array[indices])[::-1]]
        while order.size:
            current = order[0]
            keep.append(int(current))
            order = np.array(
                [
                    i
                    for i in order[1:]
                    if _rotated_iou(boxes[current], boxes[i]) < nms_thres
                ],
                dtype=np.int64,
            )
    return keep


def _restore_rrect(box, transform, family):
    """Retain source axis-wise size scaling; anisotropic OBB geometry is approximate."""
    cx, cy, w, h, angle = box
    left, top, _, _ = transform.padding
    crop_x, crop_y = transform.crop_offset
    result = [
        (cx - left) / transform.scale_x + crop_x,
        (cy - top) / transform.scale_y + crop_y,
        w / transform.scale_x,
        h / transform.scale_y,
    ]
    if family == "x5":
        oh, ow = transform.original_size
        result = [np.clip(v, 0, limit) for v, limit in zip(result, (ow, oh, ow, oh))]
    return tuple(float(v) for v in (*result, angle))


def decode_obb(
    outputs,
    contract,
    transform,
    score_thres,
    nms_thres,
    *,
    angle_sign=1.0,
    angle_offset=0.0,
    regularize=True,
    platform_family,
):
    """Return independent scalar records, preserving X5/S rotated NMS distinctions."""
    score_thres, nms_thres = float(score_thres), float(nms_thres)
    angle_sign, angle_offset = float(angle_sign), float(angle_offset)
    if not np.isfinite(score_thres) or not 0 < score_thres < 1:
        raise ValueError("score_thres must be finite in (0,1).")
    if not np.isfinite(nms_thres) or not 0 <= nms_thres <= 1:
        raise ValueError("nms_thres must be finite in [0,1].")
    if not np.isfinite(angle_sign) or not np.isfinite(angle_offset):
        raise ValueError("Angle controls must be finite.")
    if platform_family not in ("x5", "s"):
        raise ValueError("OBB requires an X5 or S platform policy.")
    ih, iw = transform.model_size
    required = {
        f"{kind}_{stride}": (1, ih // stride, iw // stride, c)
        for stride in contract.strides
        for kind, c in [("cls", contract.classes), ("box", 4), ("angle", 1)]
    }
    if set(outputs) != set(required):
        raise ValueError("OBB requires exactly its declared semantic outputs.")
    for name, shape in required.items():
        a = np.asarray(outputs[name])
        if a.shape != shape or a.dtype.kind != "f" or not np.isfinite(a).all():
            raise ValueError(f"Invalid finite floating NHWC output {name!r}.")
    boxes = []
    scores = []
    ids = []
    threshold = -np.log(1 / score_thres - 1)
    offset = np.deg2rad(angle_offset)
    for stride in contract.strides:
        classes = outputs[f"cls_{stride}"].reshape(-1, contract.classes)
        maximum = classes.max(axis=1)
        selected = maximum >= threshold
        if not selected.any():
            continue
        offsets = np.abs(outputs[f"box_{stride}"].reshape(-1, 4)[selected])
        angles = outputs[f"angle_{stride}"].reshape(-1)[selected] * angle_sign + offset
        grid = (
            np.stack(np.indices((ih // stride, iw // stride))[::-1], axis=-1)
            .reshape(-1, 2)
            .astype(np.float32)
            + 0.5
        )
        grid = grid[selected]
        left, top, right, bottom = offsets.T
        dx, dy = (right - left) / 2, (bottom - top) / 2
        c, s = np.cos(angles), np.sin(angles)
        cx = (grid[:, 0] + dx * c - dy * s) * stride
        cy = (grid[:, 1] + dx * s + dy * c) * stride
        widths = (left + right) * stride
        heights = (top + bottom) * stride
        for x, y, w, h, a, score, cid in zip(
            cx,
            cy,
            widths,
            heights,
            angles,
            sigmoid(maximum[selected]),
            np.argmax(classes[selected], axis=1),
        ):
            if regularize and w < h:
                w, h, a = h, w, a + math.pi / 2
            if platform_family == "x5":
                a = (a + math.pi / 2) % math.pi - math.pi / 2
            box = tuple(float(v) for v in (x, y, w, h, a))
            if not np.isfinite(box).all():
                raise ValueError("Decoded rotated box is nonfinite.")
            boxes.append(box)
            scores.append(float(score))
            ids.append(int(cid))
    if not boxes:
        return []
    keep = _rotated_nms(boxes, scores, ids, score_thres, nms_thres, platform_family)
    return [
        {
            "rrect": _restore_rrect(boxes[i], transform, platform_family),
            "score": scores[i],
            "id": ids[i],
        }
        for i in keep
    ]
