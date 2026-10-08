# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pure YOLOE-26 PF candidate and mask math; no DFL, NMS or dequantization.

Derived from rdk_s 380e1a2bf42041af54be6f34935e50197cfadff9. Input geometry
and kernel semantics are specific to the static 4585-class PF export.
"""

from numbers import Integral
import cv2
import numpy as np
from utils.py_utils.yoloe26_geometry import PFGeometry

CLASSES = 4585
STRIDES = (8, 16, 32)
OUTPUT_SHAPES = tuple(
    (1, 640 // stride, 640 // stride, c) for stride in STRIDES for c in (CLASSES, 4, 32)
) + (
    (1, 160, 160, 32),
)


def validate_shapes(shapes):
    """Require the fixed logical shapes, without coercing fractional dimensions."""
    shapes = tuple(tuple(shape) for shape in shapes)
    if (
        any(
            isinstance(dim, bool) or not isinstance(dim, Integral)
            for shape in shapes
            for dim in shape
        )
        or shapes != OUTPUT_SHAPES
    ):
        raise ValueError("Expected the ten 640-square YOLOE-26 PF NHWC outputs.")


def _topk_indices(values, count):
    """Select descending values with original index as the tie breaker.

    Args:
        values: Finite array flattened in row-major order.
        count: Maximum number of indices to return.

    Returns:
        A one-dimensional int64 array of flattened indices.
    """
    values = np.asarray(values).reshape(-1)
    count = min(count, values.size)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    threshold = np.partition(values, values.size - count)[values.size - count]
    higher = np.flatnonzero(values > threshold)
    tied = np.flatnonzero(values == threshold)[: count - higher.size]
    selected = np.concatenate((higher, tied))
    return selected[np.lexsort((selected, -values[selected]))]


def decode_candidates(outputs, score_threshold=0.25, max_det=300, single_label=True):
    """Match end-to-end top-k selection; never perform IoU suppression.

    Args:
        outputs: Ten NHWC float32 tensors in OUTPUT_SHAPES order: each stride
            supplies class logits, LTRB distances and coefficients, then prototype.
        score_threshold: Minimum sigmoid confidence, strictly between 0 and 1.
        max_det: Maximum detections, between 1 and 8400.
        single_label: Keep only the highest scoring class at each anchor.

    Returns:
        Boxes (N, 4) in 640x640 coordinates, scores (N,), class IDs (N,),
        and mask coefficients (N, 32), all aligned by instance.
    """
    if isinstance(max_det, bool) or not isinstance(max_det, Integral):
        raise TypeError("max_det must be an integer, not a bool or fractional value.")
    if not isinstance(single_label, bool):
        raise TypeError("single_label must be a bool.")
    if (
        not np.isfinite(score_threshold)
        or not 0 < score_threshold < 1
        or not 1 <= max_det <= 8400
    ):
        raise ValueError("Require 0 < score threshold < 1 and 1 <= max_det <= 8400")
    outputs = tuple(np.asarray(value) for value in outputs)
    validate_shapes([x.shape for x in outputs])
    if any(x.dtype != np.float32 for x in outputs):
        raise ValueError("Expected FP32 outputs for the logical decode contract")
    if any(not np.isfinite(x).all() for x in outputs):
        raise ValueError("Non-finite model output")
    # Select per-scale anchor candidates first to avoid copying the full vocabulary tensor.
    entries = []
    for scale, stride in enumerate(STRIDES):
        logits = outputs[3 * scale].reshape(-1, CLASSES)
        indices = _topk_indices(logits.max(axis=1), max_det)
        for anchor in indices:
            entries.append((float(logits[anchor].max()), scale, int(anchor)))
    entries.sort(key=lambda item: (-item[0], item[1], item[2]))
    entries = entries[:max_det]
    logits = np.stack([outputs[3 * s].reshape(-1, CLASSES)[a] for _, s, a in entries])
    if single_label:
        labels = logits.argmax(axis=1)
        anchor_ids = np.arange(len(entries))
        values = logits[anchor_ids, labels]
    else:
        selected = _topk_indices(logits, max_det)
        anchor_ids, labels = selected // CLASSES, selected % CLASSES
        values = logits.reshape(-1)[selected]
    keep = values > np.log(score_threshold / (1 - score_threshold))
    values, labels, anchor_ids = values[keep], labels[keep], anchor_ids[keep]
    boxes, coefficients = [], []
    for index in anchor_ids:
        _, scale, anchor = entries[index]
        stride = STRIDES[scale]
        grid = 640 // stride
        y, x = divmod(anchor, grid)
        left, top, right, bottom = outputs[3 * scale + 1][0, y, x]
        boxes.append(
            [
                (x + 0.5 - left) * stride,
                (y + 0.5 - top) * stride,
                (x + 0.5 + right) * stride,
                (y + 0.5 + bottom) * stride,
            ]
        )
        coefficients.append(outputs[3 * scale + 2][0, y, x])
    scores = 1 / (1 + np.exp(-np.clip(values, -80, 80)))
    with np.errstate(over="ignore", invalid="ignore"):
        box_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if not np.isfinite(box_array).all():
        raise ValueError("Decoded boxes overflowed to nonfinite values.")
    return (
        box_array,
        scores.astype(np.float32),
        labels.astype(np.int64),
        np.asarray(coefficients, dtype=np.float32).reshape(-1, 32),
    )


def restore_masks(boxes, coefficients, proto, context):
    """Restore owned F32 boxes and uint8 0/1 ROI masks with explicit geometry.

    Interpolate logits to 640, threshold at zero, crop in model space, remove
    padding, resize binary pixels with nearest interpolation, then slice each
    original-space box. Preserve empty/degenerate ROI alignment. Inverse box
    scaling uses actual rounded dimensions rather than an ideal common gain.
    """
    if not isinstance(context, PFGeometry):
        raise TypeError("Use the matching PFGeometry returned by letterbox.")
    boxes, coefficients, proto = (np.asarray(a) for a in (boxes, coefficients, proto))
    if (
        boxes.ndim != 2
        or boxes.shape[1:] != (4,)
        or coefficients.shape != (len(boxes), 32)
        or proto.shape != (160, 160, 32)
    ):
        raise ValueError(
            "Expected boxes [N,4], coefficients [N,32], prototype [160,160,32]."
        )
    if any(
        a.dtype != np.float32 or not np.isfinite(a).all()
        for a in (boxes, coefficients, proto)
    ):
        raise ValueError("Mask decoding requires finite float32 inputs.")
    if np.any(boxes[:, 2:] < boxes[:, :2]):
        raise ValueError("Box upper corners must not precede lower corners.")
    h, w = context.original_size
    rh, rw = context.resized_size
    left, top, _, _ = context.padding
    restored = boxes.copy()
    restored[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / context.scale_x, 0, w)
    restored[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / context.scale_y, 0, h)
    yy, xx = np.mgrid[:640, :640]
    masks = []
    for box, coeff, original in zip(boxes, coefficients, restored):
        raw = proto @ coeff
        if not np.isfinite(raw).all():
            raise ValueError("Mask combination overflowed to nonfinite values.")
        raw = cv2.resize(raw, (640, 640), interpolation=cv2.INTER_LINEAR)
        x1, y1, x2, y2 = box
        binary = ((raw > 0) & (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)).astype(
            np.uint8
        )
        content = binary[top : top + rh, left : left + rw]
        full = cv2.resize(content, (w, h), interpolation=cv2.INTER_NEAREST)
        x1, y1, x2, y2 = original.astype(np.int64)
        masks.append(full[y1:y2, x1:x2].copy())
    return restored, masks
