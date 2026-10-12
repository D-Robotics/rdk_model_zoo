# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Faster R-CNN MobileNetV3-Large-320-FPN two-stage detector for RDK boards.

The published deployment splits the TorchVision COCO model into two board
graphs plus exact CPU stages:

* **PartA** (``.bin`` on X5, ``.hbm`` on S) — backbone + FPN + RPN head,
  fixed ``1x3x320x320`` NV12 input, float32 outputs ``rpn_obj [3375,1]``,
  ``rpn_deltas [3375,4]`` and the single RoIAlign feature map
  ``feat0 [1,256,10,10]``.
* **CPU middle** — anchor decode, per-level top-k, sigmoid, clip,
  small-box removal, score gate, per-level NMS: at most 150 proposals,
  zero-padded to the fixed PartB batch.
* **CPU RoIAlign** — the pooler of this exact model reduces to one
  ``roi_align`` on ``feat0`` (spatial scale 1/32, 7x7 bins, 2x2 sampling,
  unaligned); proven bit-exact against the full ``MultiScaleRoIAlign``.
* **PartB** — two-layer head + class/box predictor, float32 in and out on
  every target.  X5 consumes the ``pooled [150,256,7,7]`` batch as a float
  featuremap (the graph quantizes on chip with its recorded calibration
  threshold); S consumes the packed ROI bank ``[1,256,1050,7]``.
* **CPU tail** — decode with weights (10,10,5,5), softmax, background
  removal, thresholds and per-class NMS with the exact TorchVision
  semantics (``tests/`` pins every stage against captured references).

Detection labels keep the original COCO category ids (person = 1); nothing
in this module renumbers classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Static model facts (captured from the pinned TorchVision checkpoint; see
# conversion/README.md for the provenance and tests/ for the fixtures).
# ---------------------------------------------------------------------------

INPUT_SIZE = 320                       # fixed deployment canvas
NUM_PROPOSALS = 150                    # static PartB batch == rpn post-nms top-n
NUM_CLASSES = 91                       # TorchVision slots incl. background 0
PERSON_LABEL = 1                       # original COCO category id for person

RPN_PRE_NMS_TOP_N = 150                # per FPN level
RPN_POST_NMS_TOP_N = 150
RPN_NMS_THRESH = 0.7
RPN_SCORE_THRESH = 0.05
RPN_MIN_SIZE = 0.001
RPN_BOX_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

BOX_SCORE_THRESH = 0.05
BOX_NMS_THRESH = 0.5
BOX_DETECTIONS_PER_IMG = 100
BOX_MIN_SIZE = 1e-2
BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
BBOX_XFORM_CLIP = float(np.log(1000.0 / 16.0))

POOL_SPATIAL_SCALE = 1.0 / 32.0        # feat0 stride; proven exact for this model
POOL_OUTPUT_SIZE = 7
POOL_SAMPLING_RATIO = 2

ANCHOR_SIZES = (32, 64, 128, 256, 512)  # identical tuple on every FPN level
ANCHOR_RATIOS = (0.5, 1.0, 2.0)
ANCHOR_LEVEL_GRIDS = ((10, 10), (10, 10), (5, 5))   # feat0, feat1, pool
ANCHOR_LEVEL_STRIDES = (32, 32, 64)
NUM_ANCHORS_PER_LEVEL = tuple(h * w * len(ANCHOR_SIZES) * len(ANCHOR_RATIOS)
                              for h, w in ANCHOR_LEVEL_GRIDS)  # (1500,1500,375)

PARTA_OUTPUT_NAMES = ("rpn_obj", "rpn_deltas", "feat0")
PARTB_OUTPUT_NAMES = ("cls_packed", "reg_packed")          # S packed layout
PARTB_OUTPUT_NAMES_BATCH = ("class_logits", "box_regression")  # X5 batch150 layout

#: PartB input layout per board family.
PARTB_LAYOUT_PACKED = "packed"    # S: float32 ROI bank [1,256,1050,7]
PARTB_LAYOUT_POOLED = "pooled"    # X5: float32 pooled batch [150,256,7,7]


# ---------------------------------------------------------------------------
# Anchor generation (TorchVision AnchorGenerator port; pinned by tests).
# ---------------------------------------------------------------------------

def generate_anchors(image_size: int = INPUT_SIZE) -> np.ndarray:
    """Return the ``(3375, 4)`` float32 anchor table in TorchVision order.

    Cell anchors are zero-centered boxes with widths ``size / sqrt(ratio)``
    and heights ``size * sqrt(ratio)`` (ratio outer, size inner), rounded to
    integers; grid shifts place each cell anchor center at
    ``(0.5 + j) * stride`` positions, cells row-major; levels concatenate in
    feature order.
    """
    ratios = np.sqrt(np.asarray(ANCHOR_RATIOS, dtype=np.float64))
    sizes = np.asarray(ANCHOR_SIZES, dtype=np.float64)
    # TorchVision ordering: ratio index outer, size index inner.
    ws = (sizes[None, :] / ratios[:, None]).reshape(-1)
    hs = (sizes[None, :] * ratios[:, None]).reshape(-1)
    base = np.round(np.stack([-ws, -hs, ws, hs], axis=1) / 2.0)  # (15, 4)
    levels = []
    for (gh, gw), stride in zip(ANCHOR_LEVEL_GRIDS, ANCHOR_LEVEL_STRIDES):
        # TorchVision derives strides from image_size // grid; assert matches.
        assert image_size // gh == stride, (image_size, gh, stride)
        shifts_x = (np.arange(gw, dtype=np.int64) * stride)
        shifts_y = (np.arange(gh, dtype=np.int64) * stride)
        sx, sy = np.meshgrid(shifts_x, shifts_y, indexing="xy")  # row-major
        sx = sx.reshape(-1, 1)
        sy = sy.reshape(-1, 1)
        shifts = np.concatenate((sx, sy, sx, sy), axis=1)        # (G, 4)
        levels.append((shifts[:, None, :] + base[None, :, :]).reshape(-1, 4))
    return np.concatenate(levels, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Geometry: exact TorchVision box ops (no +1 area convention anywhere).
# ---------------------------------------------------------------------------

def decode_deltas(anchor_xyxy: np.ndarray, deltas: np.ndarray,
                  weights: Sequence[float]) -> np.ndarray:
    """Decode RPN/tail deltas exactly like ``BoxCoder.decode_single``.

    ``deltas`` may be ``(N, 4)`` against ``(N, 4)`` anchors (RPN) or
    ``(N, 4*K)`` against ``(N, 4)`` anchors (tail predictor, ``K = 91``);
    ``dw``/``dh`` clamp at ``log(1000/16)`` before the exponential.
    """
    anchors = np.asarray(anchor_xyxy, dtype=np.float32)
    codes = np.asarray(deltas, dtype=np.float32)
    wx, wy, ww, wh = (float(w) for w in weights)
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights
    dx = codes[:, 0::4] / wx
    dy = codes[:, 1::4] / wy
    dw = np.minimum(codes[:, 2::4] / ww, BBOX_XFORM_CLIP)
    dh = np.minimum(codes[:, 3::4] / wh, BBOX_XFORM_CLIP)
    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = np.exp(dw) * widths[:, None]
    pred_h = np.exp(dh) * heights[:, None]
    half_w = np.float32(0.5) * pred_w
    half_h = np.float32(0.5) * pred_h
    out = np.empty((codes.shape[0], codes.shape[1]), dtype=np.float32)
    out[:, 0::4] = pred_ctr_x - half_w
    out[:, 1::4] = pred_ctr_y - half_h
    out[:, 2::4] = pred_ctr_x + half_w
    out[:, 3::4] = pred_ctr_y + half_h
    return out


def clip_boxes_to_image(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """Clamp xyxy boxes into ``[0, height] x [0, width]`` (TorchVision).

    Operates on the last axis with ``[..., 0::2]`` / ``[..., 1::2]`` so both
    ``(N, 4)`` proposal arrays and the ``(N, 91, 4)`` per-class tail arrays
    clip coordinates, never leading axes.  TorchVision clamps continuous
    coordinates to the full width/height, not ``size - 1``.
    """
    out = np.array(boxes, dtype=np.float32, copy=True)
    out[..., 0::2] = np.clip(out[..., 0::2], 0.0, float(width))
    out[..., 1::2] = np.clip(out[..., 1::2], 0.0, float(height))
    return out


def remove_small_boxes(boxes: np.ndarray, min_size: float) -> np.ndarray:
    """Return indices of boxes whose width and height both reach ``min_size``."""
    keep = ((boxes[:, 2] - boxes[:, 0]) >= min_size) & \
           ((boxes[:, 3] - boxes[:, 1]) >= min_size)
    return np.flatnonzero(keep)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS with TorchVision semantics: suppress IoU strictly greater.

    Boxes are visited in descending score order (stable for equal scores);
    intersection and union areas use continuous coordinates without the
    ``+1`` pixel convention.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep: list[int] = []
    for pos in range(order.shape[0]):
        i = order[pos]
        if not np.isfinite(boxes[i]).all() or areas[i] <= 0:
            continue
        suppressed = False
        for j in keep:
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            iw = max(0.0, float(xx2 - xx1))
            ih = max(0.0, float(yy2 - yy1))
            inter = iw * ih
            union = areas[i] + areas[j] - inter
            if union > 0 and inter / union > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, idxs: np.ndarray,
                iou_threshold: float) -> np.ndarray:
    """Per-group NMS via the TorchVision coordinate-offset composition."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.empty(0, dtype=np.int64)
    max_coordinate = float(boxes.max())
    offsets = np.asarray(idxs, dtype=np.float64) * (max_coordinate + 1.0)
    return nms(boxes + offsets[:, None].astype(np.float32), scores, iou_threshold)


# ---------------------------------------------------------------------------
# CPU stage 1: RPN selection.
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise sigmoid on float32."""
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def rpn_select(rpn_obj: np.ndarray, rpn_deltas: np.ndarray,
               anchors: np.ndarray, image_size: int = INPUT_SIZE) -> tuple[np.ndarray, int]:
    """Select at most 150 proposals; return ``(padded [150,4], n_valid)``.

    Mirrors ``RegionProposalNetwork.filter_proposals``: per-level top-150 on
    the raw objectness logits, sigmoid after selection, clip, small-box
    removal (0.001), ``score >= 0.05`` gate, per-level NMS (0.7), global
    top-150 by score.  Padding rows are zeros and never reach the tail.
    """
    obj = np.asarray(rpn_obj, dtype=np.float32).reshape(-1)
    deltas = np.asarray(rpn_deltas, dtype=np.float32).reshape(-1, 4)
    if obj.shape[0] != anchors.shape[0] or deltas.shape[0] != anchors.shape[0]:
        raise ValueError(
            f"PartA RPN outputs must match {anchors.shape[0]} anchors, got "
            f"{obj.shape[0]} objectness and {deltas.shape[0]} delta rows.")
    if not np.isfinite(obj).all() or not np.isfinite(deltas).all():
        raise ValueError("PartA RPN outputs contain non-finite values.")

    decoded = decode_deltas(anchors, deltas, RPN_BOX_WEIGHTS)

    boxes_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    level_parts: list[np.ndarray] = []
    offset = 0
    for level, count in enumerate(NUM_ANCHORS_PER_LEVEL):
        ob = obj[offset:offset + count]
        k = min(RPN_PRE_NMS_TOP_N, count)
        # Top-k on raw logits == top-k on sigmoid (monotone); stable ties.
        top = np.argsort(-ob, kind="stable")[:k]
        top += offset
        boxes_parts.append(decoded[top])
        score_parts.append(_sigmoid(obj[top]))
        level_parts.append(np.full(k, level, dtype=np.int64))
        offset += count

    boxes = np.concatenate(boxes_parts, axis=0)
    scores = np.concatenate(score_parts, axis=0)
    levels = np.concatenate(level_parts, axis=0)

    boxes = clip_boxes_to_image(boxes, image_size, image_size)
    keep = remove_small_boxes(boxes, RPN_MIN_SIZE)
    boxes, scores, levels = boxes[keep], scores[keep], levels[keep]
    keep = np.flatnonzero(scores >= RPN_SCORE_THRESH)
    boxes, scores, levels = boxes[keep], scores[keep], levels[keep]
    keep = batched_nms(boxes, scores, levels, RPN_NMS_THRESH)
    keep = keep[:RPN_POST_NMS_TOP_N]
    boxes, scores = boxes[keep], scores[keep]

    n = int(boxes.shape[0])
    padded = np.zeros((NUM_PROPOSALS, 4), dtype=np.float32)
    if n:
        if n > NUM_PROPOSALS:  # defensive: nms kept <= 150 already
            boxes, scores, n = boxes[:NUM_PROPOSALS], scores[:NUM_PROPOSALS], NUM_PROPOSALS
        padded[:n] = boxes
    return padded, n


# ---------------------------------------------------------------------------
# CPU stage 2: RoIAlign on the single consumed feature map.
# ---------------------------------------------------------------------------

def roi_align_feat0(feat0: np.ndarray, rois: np.ndarray,
                    spatial_scale: float = POOL_SPATIAL_SCALE,
                    output_size: int = POOL_OUTPUT_SIZE,
                    sampling_ratio: int = POOL_SAMPLING_RATIO) -> np.ndarray:
    """RoIAlign ``feat0 [C,H,W]`` with ``rois [N,4]`` (unaligned, TorchVision).

    Each 7x7 output bin averages ``sampling_ratio^2`` bilinear samples taken
    at ``((q + 0.5) * bin_size / sampling_ratio)`` offsets inside the bin.
    Samples outside ``[-1, H]`` (or ``W``) contribute zero; otherwise the
    bilinear corners clamp to the feature border, matching the reference
    kernel exactly.  The kernel's malformed-ROI clamp applies: any ROI
    narrower or shorter than one feature pixel (``1 / spatial_scale``
    image pixels) pools over a forced ``1x1`` feature-pixel window — the
    exact TorchVision ``std::max(roi_width, 1)`` semantics that degenerate
    and zero-area boxes hit.
    """
    feat = np.asarray(feat0, dtype=np.float32)
    rois = np.asarray(rois, dtype=np.float32)
    if feat.ndim != 3:
        raise ValueError(f"feat0 must be [C,H,W], got shape {feat.shape}.")
    channels, height, width = feat.shape
    n = rois.shape[0]
    out = np.zeros((n, channels, output_size, output_size), dtype=np.float32)
    if n == 0:
        return out
    inv = 1.0 / (sampling_ratio * sampling_ratio)

    x1 = rois[:, 0] * spatial_scale
    y1 = rois[:, 1] * spatial_scale
    x2 = rois[:, 2] * spatial_scale
    y2 = rois[:, 3] * spatial_scale
    # TorchVision forces malformed ROIs to 1x1 in feature pixels.
    roi_w = np.maximum(x2 - x1, 1.0)
    roi_h = np.maximum(y2 - y1, 1.0)
    bin_w = roi_w / output_size
    bin_h = roi_h / output_size

    for iy in range(output_size):
        for ix in range(output_size):
            # Sample offsets inside one bin (unaligned: no half-pixel shift).
            frac_y = ((np.arange(sampling_ratio, dtype=np.float32) + 0.5)
                      / sampling_ratio)
            frac_x = ((np.arange(sampling_ratio, dtype=np.float32) + 0.5)
                      / sampling_ratio)
            yb = y1 + iy * bin_h
            xb = x1 + ix * bin_w
            acc = np.zeros((n, channels), dtype=np.float64)
            for fy in frac_y:
                ys = yb + fy * bin_h
                for fx in frac_x:
                    xs = xb + fx * bin_w
                    acc += _bilinear_batch(feat, ys, xs, height, width)
            out[:, :, iy, ix] = (acc * inv).astype(np.float32)
    return out


def _bilinear_batch(feat: np.ndarray, ys: np.ndarray, xs: np.ndarray,
                    height: int, width: int) -> np.ndarray:
    """Bilinear sample ``feat [C,H,W]`` at per-ROI ``(ys, xs)`` centers.

    TorchVision kernel semantics: coordinates with ``y < -1`` or ``y > H``
    (or the x equivalents) yield zero; otherwise negatives clamp to 0, the
    low/high neighbour pair clamps to the last feature row/column (with the
    fractional part zeroed there), and standard bilinear weights apply.
    """
    n = ys.shape[0]
    valid = (ys >= -1.0) & (ys <= height) & (xs >= -1.0) & (xs <= width)
    result = np.zeros((n, feat.shape[0]), dtype=np.float64)
    if not valid.any():
        return result
    y = np.maximum(np.where(valid, ys, 0.0), 0.0)
    x = np.maximum(np.where(valid, xs, 0.0), 0.0)
    y_low = y.astype(np.int64)
    x_low = x.astype(np.int64)
    clamp_y = y_low >= height - 1
    clamp_x = x_low >= width - 1
    y_low = np.where(clamp_y, height - 1, y_low)
    x_low = np.where(clamp_x, width - 1, x_low)
    y_high = np.where(clamp_y, height - 1, y_low + 1)
    x_high = np.where(clamp_x, width - 1, x_low + 1)
    y_used = np.where(clamp_y, y_low.astype(np.float64), y)
    x_used = np.where(clamp_x, x_low.astype(np.float64), x)
    ly = y_used - y_low
    lx = x_used - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx

    f = feat.astype(np.float64)
    v00 = f[:, y_low, x_low]
    v01 = f[:, y_low, x_high]
    v10 = f[:, y_high, x_low]
    v11 = f[:, y_high, x_high]
    w00 = (hy * hx)[None, :]
    w01 = (hy * lx)[None, :]
    w10 = (ly * hx)[None, :]
    w11 = (ly * lx)[None, :]
    vals = (v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11).T  # (n, C)
    result[valid] = vals[valid]
    return result


# ---------------------------------------------------------------------------
# PartB input preparation for both board layouts.
# ---------------------------------------------------------------------------

def pack_bank(pooled: np.ndarray) -> np.ndarray:
    """Re-layout ``pooled [150,256,7,7]`` into the S ROI bank tensor."""
    pooled = np.asarray(pooled, dtype=np.float32)
    expected = (NUM_PROPOSALS, 256, POOL_OUTPUT_SIZE, POOL_OUTPUT_SIZE)
    if pooled.shape != expected:
        raise ValueError(f"pooled must be {expected}, got {pooled.shape}")
    bank = pooled.transpose(1, 0, 2, 3).reshape(1, 256, NUM_PROPOSALS * POOL_OUTPUT_SIZE,
                                                POOL_OUTPUT_SIZE)
    return np.ascontiguousarray(bank)


def unpack_bank_outputs(cls_packed: np.ndarray, reg_packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transpose S packed outputs ``[1,C,150,1]`` into ``[150,C]`` rows."""
    cls = np.asarray(cls_packed, dtype=np.float32).reshape(1, -1, NUM_PROPOSALS, 1)[0, :, :, 0].T
    reg = np.asarray(reg_packed, dtype=np.float32).reshape(1, -1, NUM_PROPOSALS, 1)[0, :, :, 0].T
    return np.ascontiguousarray(cls), np.ascontiguousarray(reg)


# ---------------------------------------------------------------------------
# CPU stage 3: final classification tail.
# ---------------------------------------------------------------------------

def tail(class_logits: np.ndarray, box_regression: np.ndarray,
         proposals: np.ndarray, n_valid: int,
         image_size: int = INPUT_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode PartB outputs into ``(boxes, scores, labels)`` canvas boxes.

    Exact ``postprocess_detections`` order: decode (weights 10,10,5,5),
    softmax, clip, drop background column 0, flatten roi-major/class-minor,
    strict ``score > 0.05`` filter, small-box removal (1e-2), per-class NMS
    (0.5), top 100.  Only the first ``n_valid`` rows participate.
    """
    logits = np.asarray(class_logits, dtype=np.float32)[:n_valid]
    reg = np.asarray(box_regression, dtype=np.float32)[:n_valid]
    prop = np.asarray(proposals, dtype=np.float32)[:n_valid]
    if logits.shape != (prop.shape[0], NUM_CLASSES) or \
            reg.shape != (prop.shape[0], NUM_CLASSES * 4):
        raise ValueError(
            f"PartB outputs must be [N,91] and [N,364] for {prop.shape[0]} rows, "
            f"got {logits.shape} and {reg.shape}.")
    if not np.isfinite(logits).all() or not np.isfinite(reg).all():
        raise ValueError("PartB outputs contain non-finite values.")

    if prop.shape[0] == 0:
        empty4 = np.empty((0, 4), dtype=np.float32)
        return empty4, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

    boxes = decode_deltas(prop, reg, BOX_WEIGHTS).reshape(prop.shape[0], NUM_CLASSES, 4)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    boxes = clip_boxes_to_image(boxes, image_size, image_size)[:, 1:, :].reshape(-1, 4)
    scores = probs[:, 1:].reshape(-1)  # drop background column, flatten roi-major
    labels = np.tile(np.arange(1, NUM_CLASSES, dtype=np.int64), prop.shape[0])

    keep = np.flatnonzero(scores > BOX_SCORE_THRESH)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keep = remove_small_boxes(boxes, BOX_MIN_SIZE)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keep = batched_nms(boxes, scores, labels, BOX_NMS_THRESH)
    keep = keep[:BOX_DETECTIONS_PER_IMG]
    return boxes[keep].copy(), scores[keep].copy(), labels[keep].copy()


# ---------------------------------------------------------------------------
# Letterbox geometry shared by preprocess/postprocess.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LetterboxContext:
    """Immutable per-call geometry for one deployment canvas.

    ``scale`` is the nominal aspect-preserving factor; the resize targets
    ``net_size`` = ``round(dim * scale)`` per axis, so the achieved per-axis
    factors differ from the nominal one by rounding.  ``to_original`` divides
    by the achieved factors (exactly inverting the resize), then clips to
    the source size.  Padding is bottom/right only.
    """
    original_size: tuple[int, int]        # (height, width) of the source image
    net_size: tuple[int, int]             # letterboxed content size (<= 320)
    canvas_size: int
    scale: float

    @property
    def scale_y(self) -> float:
        """Achieved vertical factor: net height over source height."""
        return self.net_size[0] / self.original_size[0]

    @property
    def scale_x(self) -> float:
        """Achieved horizontal factor: net width over source width."""
        return self.net_size[1] / self.original_size[1]

    def to_original(self, boxes: np.ndarray) -> np.ndarray:
        """Map canvas xyxy boxes to original-image pixels (clipped)."""
        out = np.array(boxes, dtype=np.float32, copy=True)
        out[..., 0::2] = out[..., 0::2] / np.float32(self.scale_x)
        out[..., 1::2] = out[..., 1::2] / np.float32(self.scale_y)
        h, w = self.original_size
        return clip_boxes_to_image(out, h, w)


def letterbox_geometry(height: int, width: int,
                       canvas_size: int = INPUT_SIZE) -> LetterboxContext:
    """Compute the letterbox geometry for one source image size."""
    if height < 1 or width < 1:
        raise ValueError(f"image size must be positive, got {height}x{width}.")
    scale = min(canvas_size / width, canvas_size / height)
    net_h = int(round(height * scale))
    net_w = int(round(width * scale))
    if net_h < 1 or net_w < 1:
        raise ValueError("image aspect ratio produces an empty letterbox dimension.")
    return LetterboxContext((height, width), (net_h, net_w), canvas_size, float(scale))


# ---------------------------------------------------------------------------
# Board-side pipeline objects.
# ---------------------------------------------------------------------------

class ContractError(ValueError):
    """The selected pair or observed runtime metadata breaks the contract."""


@dataclass(frozen=True)
class PairedSelection:
    """Resolved PartA/PartB pair for one target plus its static contract.

    ``parta_sha256``/``partb_sha256`` are the descriptor-recorded digests of
    the verified artifacts — ``None`` for a custom ``--parta-path/--partb-path``
    pair.  The digests gate artifact verification: a file published under a
    recorded name must hash to the recorded bytes before it runs.
    """
    target: str
    parta_path: Any                # Path-like; kept loose for host tests
    partb_path: Any
    partb_layout: str              # PARTB_LAYOUT_PACKED | PARTB_LAYOUT_POOLED
    asset_id: str = "fasterrcnn"
    parta_sha256: str | None = None
    partb_sha256: str | None = None

    def __post_init__(self):
        if self.partb_layout not in (PARTB_LAYOUT_PACKED, PARTB_LAYOUT_POOLED):
            raise ValueError(f"unknown PartB layout {self.partb_layout!r}.")


@dataclass(frozen=True)
class PreparedInput:
    """Owned NV12 tensors for PartA plus the per-call geometry context."""
    tensors: dict[str, np.ndarray]
    context: LetterboxContext


@dataclass(frozen=True)
class Detection:
    """One detection in original-image pixels with the original COCO id."""
    label: int
    score: float
    box: tuple[float, float, float, float]


class DetectionResult:
    """Owned arrays: float32 xyxy boxes, float32 scores, int64 COCO labels.

    The constructor copies its inputs, so mutating the caller's arrays after
    construction (or reusing the same buffers for the next frame) can never
    change a published result.
    """

    def __init__(self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray):
        self.boxes = np.array(boxes, dtype=np.float32, copy=True).reshape(-1, 4)
        self.scores = np.array(scores, dtype=np.float32, copy=True).reshape(-1)
        self.labels = np.array(labels, dtype=np.int64, copy=True).reshape(-1)
        if not (self.boxes.shape[0] == self.scores.shape[0] == self.labels.shape[0]):
            raise ValueError("boxes/scores/labels lengths disagree.")

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    def __iter__(self):
        return iter([Detection(int(l), float(s), tuple(float(v) for v in b))
                     for l, s, b in zip(self.labels, self.scores, self.boxes)])

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.boxes, self.scores, self.labels

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DetectionResult):
            return NotImplemented
        return (np.array_equal(self.boxes, other.boxes)
                and np.array_equal(self.scores, other.scores)
                and np.array_equal(self.labels, other.labels))


def _empty_result() -> DetectionResult:
    return DetectionResult(np.empty((0, 4), dtype=np.float32),
                           np.empty(0, dtype=np.float32),
                           np.empty(0, dtype=np.int64))


def nv12_tensors(canvas_bgr: np.ndarray, target: str) -> dict[str, np.ndarray]:
    """Encode a BGR uint8 canvas as the target's NV12 input tensors.

    The shared converter (``utils.py_utils.image.bgr_to_nv12_planes``)
    expects **BGR** pixels and emits limited-range BT.601 planes — the
    colorspace the compiled PartA input declares (``bt601_video``).  X5
    takes one flat packed buffer; S takes the split Y and UV planes.
    """
    from utils.py_utils.image import bgr_to_nv12_planes

    if (not isinstance(canvas_bgr, np.ndarray) or canvas_bgr.dtype != np.uint8
            or canvas_bgr.ndim != 3 or canvas_bgr.shape[2] != 3):
        raise ValueError("Expected an HWC uint8 BGR canvas.")
    y, uv = bgr_to_nv12_planes(canvas_bgr)
    if target == "x5":
        return {"images": np.concatenate((y.reshape(-1), uv.reshape(-1)))}
    return {"images_y": y, "images_uv": uv}


class PairedRuntime:
    """Two board model sessions with validated tensor plumbing.

    ``runtime_factory(path) -> runtime`` is the only board entry point; the
    default factory uses the shared ``utils.py_utils.runtime`` session which
    resolves ``hbm_runtime`` lazily and gates the board identity.  PartA is
    fed NV12 (packed flat bytes on X5, split Y/UV planes on S); PartB is fed
    the layout its artifact declares, always in float32.  After each model
    loads, the live SDK object's public tensor metadata (names, shapes,
    dtypes) is read — through the ``RuntimeSession`` wrapper, off the
    underlying ``.runtime`` SDK object — and validated against the recorded
    contract, so a wrong artifact fails with a named error instead of
    feeding bytes to the wrong layout.
    """

    def __init__(self, selection: PairedSelection,
                 runtime_factory: Callable[[Any], Any] | None = None):
        self.selection = selection
        self._factory = runtime_factory
        self.parta: Any = None
        self.partb: Any = None

    # -- loading ---------------------------------------------------------
    def load(self) -> "PairedRuntime":
        if self.parta is None:
            self.parta = self._open(self.selection.parta_path)
            self._validate_live(self.parta, "parta")
        if self.partb is None:
            self.partb = self._open(self.selection.partb_path)
            self._validate_live(self.partb, "partb")
        return self

    def _open(self, path) -> Any:
        factory = self._factory
        if factory is None:
            from utils.py_utils.runtime import RuntimeSession
            session = RuntimeSession(path, target=self.selection.target)
            session.load()
            return session
        return factory(path)

    def _validate_live(self, session: Any, which: str) -> None:
        """Validate the live SDK tensor metadata for one loaded model.

        ``RuntimeSession`` exposes only ``runtime``/``loaded``/``run``/``load``;
        the tensor facts live on the wrapped SDK object, so the wrapper is
        unwrapped before reading.
        """
        from utils.py_utils.runtime_meta import (
            MetadataMismatchError,
            RuntimeMetadata,
        )

        sdk = _unwrap_sdk(session)
        try:
            meta = RuntimeMetadata.from_runtime(sdk)
        except MetadataMismatchError as exc:
            raise ContractError(
                f"{which} artifact did not expose live tensor metadata "
                f"({self.selection.partb_path if which == 'partb' else self.selection.parta_path}): {exc}"
            ) from exc
        contract = LIVE_TENSOR_CONTRACTS[self.selection.target][which]
        _check_live_contract(meta, contract, which)

    # -- execution -------------------------------------------------------
    def run_parta(self, tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs = _run_model(self.parta, tensors)
        expected = {"rpn_obj": (3375, 1), "rpn_deltas": (3375, 4), "feat0": (1, 256, 10, 10)}
        out: dict[str, np.ndarray] = {}
        for name, shape in expected.items():
            if name not in outputs:
                raise ContractError(f"PartA output {name!r} missing; got {sorted(outputs)}.")
            arr = np.asarray(outputs[name], dtype=np.float32)
            if arr.size != int(np.prod(shape)):
                raise ContractError(f"PartA output {name!r} has {arr.size} values, expected {shape}.")
            out[name] = arr.reshape(shape)
        return out

    def run_partb(self, pooled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pooled = np.asarray(pooled, dtype=np.float32)
        if self.selection.partb_layout == PARTB_LAYOUT_PACKED:
            outputs = _run_model(self.partb, {"bank": pack_bank(pooled)})
            _require_outputs(outputs, PARTB_OUTPUT_NAMES, "PartB")
            cls_arr, reg_arr = outputs["cls_packed"], outputs["reg_packed"]
            if cls_arr.size != NUM_PROPOSALS * NUM_CLASSES or \
                    reg_arr.size != NUM_PROPOSALS * NUM_CLASSES * 4:
                raise ContractError(
                    f"PartB packed outputs have {cls_arr.size}/{reg_arr.size} values, "
                    f"expected {NUM_PROPOSALS}x{NUM_CLASSES} and "
                    f"{NUM_PROPOSALS}x{NUM_CLASSES * 4}.")
            cls, reg = unpack_bank_outputs(cls_arr, reg_arr)
        else:
            outputs = _run_model(self.partb, {"pooled": pooled})
            _require_outputs(outputs, PARTB_OUTPUT_NAMES_BATCH, "PartB")
            cls = np.asarray(outputs["class_logits"], dtype=np.float32).reshape(NUM_PROPOSALS, NUM_CLASSES)
            reg = np.asarray(outputs["box_regression"], dtype=np.float32).reshape(NUM_PROPOSALS, NUM_CLASSES * 4)
        if cls.shape != (NUM_PROPOSALS, NUM_CLASSES) or \
                reg.shape != (NUM_PROPOSALS, NUM_CLASSES * 4):
            raise ContractError(f"PartB outputs have wrong shapes {cls.shape}/{reg.shape}.")
        return cls, reg


def _unwrap_sdk(session: Any) -> Any:
    """The underlying board SDK object behind a ``RuntimeSession`` wrapper.

    The wrapper deliberately exposes only ``runtime``/``loaded`` and
    ``run``/``load``; tensor metadata (``model_names``, ``input_names``,
    ``input_shapes``, …) lives on the SDK object it wraps.  Objects that
    are not wrappers (injected host doubles) pass through unchanged.
    """
    inner = getattr(session, "runtime", None)
    return session if inner is None else inner


def _require_outputs(outputs: Mapping[str, Any], names: Sequence[str],
                     what: str) -> None:
    missing = [name for name in names if name not in outputs]
    if missing:
        raise ContractError(f"{what} outputs {missing} missing; got {sorted(outputs)}.")


#: Live SDK tensor contract per target, read from the loaded artifacts'
#: compile records and the live board SDK metadata.  Inputs:
#: ``name -> (allowed dtypes, shape_spec)`` where a tuple shape spec must
#: match after trailing-singleton normalization and an int spec is an
#: element count.  Two board spellings are normalized, never guessed:
#: the X5 packed-NV12 input (flat byte buffer or the ``(1,3,320,320)``
#: image layout, both accepted via :class:`_ShapeAlternatives`) and the X5
#: packed-model virtual shapes, which pad the graph-declared shape to 4D
#: with trailing singleton dimensions (``(3375,1)`` spelled
#: ``(3375,1,1,1)``); the flat buffer and element count are unchanged.
#: Outputs: ``name -> (dtype, shape)`` compared with the same rule.
class _ShapeAlternatives:
    """A shape spec where any one of ``specs`` may match."""

    def __init__(self, *specs: int | tuple):
        self.specs = tuple(specs)


def _normalize_shape(shape: tuple) -> tuple:
    """Trim trailing singleton dimensions from one SDK-reported shape.

    The X5 packed-model spelling pads graph-declared shapes to 4D with
    trailing 1s; a trailing size-1 dimension contributes exactly one
    element, so the normalized shapes (and the flat row-major buffer)
    are identical.  At least one dimension always remains.
    """
    dims = list(shape)
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return tuple(dims)


LIVE_TENSOR_CONTRACTS: dict[str, dict[str, dict]] = {
    "x5": {
        "parta": {
            "inputs": {"images": (("uint8", "nv12"), _ShapeAlternatives(
                INPUT_SIZE * INPUT_SIZE * 3 // 2,   # flat packed-NV12 bytes
                (1, 3, INPUT_SIZE, INPUT_SIZE),     # image-layout spelling
            ))},
            "outputs": {
                "rpn_obj": ("float32", (3375, 1)),
                "rpn_deltas": ("float32", (3375, 4)),
                "feat0": ("float32", (1, 256, 10, 10)),
            },
        },
        "partb": {
            "inputs": {"pooled": (("float32",), (NUM_PROPOSALS, 256, 7, 7))},
            "outputs": {
                "class_logits": ("float32", (NUM_PROPOSALS, NUM_CLASSES)),
                "box_regression": ("float32", (NUM_PROPOSALS, NUM_CLASSES * 4)),
            },
        },
    },
    "s": {
        "parta": {
            "inputs": {
                "images_y": (("uint8", "nv12"), (1, INPUT_SIZE, INPUT_SIZE, 1)),
                "images_uv": (("uint8", "nv12"), (1, INPUT_SIZE // 2, INPUT_SIZE // 2, 2)),
            },
            "outputs": {
                "rpn_obj": ("float32", (3375, 1)),
                "rpn_deltas": ("float32", (3375, 4)),
                "feat0": ("float32", (1, 256, 10, 10)),
            },
        },
        "partb": {
            "inputs": {"bank": (("float32",), (1, 256, NUM_PROPOSALS * 7, 7))},
            "outputs": {
                "cls_packed": ("float32", (1, NUM_CLASSES, NUM_PROPOSALS, 1)),
                "reg_packed": ("float32", (1, NUM_CLASSES * 4, NUM_PROPOSALS, 1)),
            },
        },
    },
}
for _target in ("s100", "s100p", "s600"):
    LIVE_TENSOR_CONTRACTS[_target] = LIVE_TENSOR_CONTRACTS["s"]


def _check_live_contract(meta: Any, contract: dict, which: str) -> None:
    """Compare live SDK tensor facts with the recorded contract.

    The required facts — names, dtypes and shapes of every contracted input
    and output — must be present and match: an SDK object that clears them
    fails the check instead of slipping through, including for custom
    ``--parta-path``/``--partb-path`` pairs.
    """
    inputs: dict = contract["inputs"]
    names = tuple(meta.input_names or ())
    if set(names) != set(inputs):
        raise ContractError(
            f"{which} live inputs {names!r} != contract "
            f"{sorted(inputs)}; the artifact does not match this sample.")
    for name, (dtypes, shape_spec) in inputs.items():
        got_dtype = meta.input_dtypes.get(name)
        if got_dtype is None:
            raise ContractError(
                f"{which} input {name!r} exposes no dtype; the required facts "
                "must be present for a bound pair.")
        if got_dtype not in dtypes:
            raise ContractError(
                f"{which} input {name!r} dtype {got_dtype!r} not in {list(dtypes)}.")
        got_shape = tuple(meta.input_shapes.get(name) or ())
        if not got_shape:
            raise ContractError(
                f"{which} input {name!r} exposes no shape; the required facts "
                "must be present for a bound pair.")
        _require_shape_spec(got_shape, shape_spec, f"{which} input {name!r}")
    outputs: dict = contract["outputs"]
    out_names = tuple(meta.output_names or ())
    if set(out_names) != set(outputs):
        raise ContractError(
            f"{which} live outputs {out_names!r} != contract "
            f"{sorted(outputs)}; the artifact does not match this sample.")
    for name, (dtype, shape) in outputs.items():
        got_dtype = meta.output_dtypes.get(name)
        if got_dtype is None:
            raise ContractError(
                f"{which} output {name!r} exposes no dtype; the required facts "
                "must be present for a bound pair.")
        if got_dtype != dtype:
            raise ContractError(
                f"{which} output {name!r} dtype {got_dtype!r} != {dtype}.")
        got_shape = tuple(meta.output_shapes.get(name) or ())
        if not got_shape:
            raise ContractError(
                f"{which} output {name!r} exposes no shape; the required facts "
                "must be present for a bound pair.")
        if _normalize_shape(got_shape) != _normalize_shape(tuple(shape)):
            raise ContractError(
                f"{which} output {name!r} shape {got_shape} != {shape}.")


def _require_shape_spec(got_shape: tuple, spec: int | tuple | _ShapeAlternatives,
                        what: str) -> None:
    """Match one shape spec: an int is an element count, a tuple an exact
    layout (after trailing-singleton normalization), and
    :class:`_ShapeAlternatives` a set of accepted spellings."""
    specs = spec.specs if isinstance(spec, _ShapeAlternatives) else (spec,)
    for one in specs:
        if isinstance(one, int):
            if int(np.prod(got_shape)) == one:
                return
        elif _normalize_shape(got_shape) == _normalize_shape(tuple(one)):
            return
    raise ContractError(
        f"{what} shape {got_shape} does not match the declared contract "
        f"{list(specs)}.")


def file_sha256(path: Any) -> str:
    """Stream a file's sha256 hex digest (board artifacts reach ~30 MB)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_model(model: Any, tensors: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
    """Run one model; return its output-name mapping.

    ``hbm_runtime.run`` returns ``{model_name: {output_name: array}}`` (pybind
    ``dict[str, dict[str, numpy.ndarray]]`` — the same envelope the yolo
    sample unwraps by model name).
    These artifacts each carry exactly one model, so the single-model envelope
    is unwrapped to the inner mapping; a mapping already keyed by output names
    (host doubles, or a future SDK spelling) passes through unchanged.
    """
    if hasattr(model, "run"):
        result = model.run(tensors)
    else:  # pragma: no cover - injected host doubles
        result = model(tensors)
    if not isinstance(result, Mapping):
        raise ContractError("runtime must return a mapping of named outputs.")
    names = tuple(getattr(_unwrap_sdk(model), "model_names", ()) or ())
    if len(names) == 1 and isinstance(result.get(names[0]), Mapping):
        return result[names[0]]
    return result


class FasterRCNNTask:
    """Run one bound Faster R-CNN pair end to end.

    Input is a non-empty BGR ``uint8`` ``(H,W,3)`` array.  ``preprocess``
    letterboxes to the 320x320 canvas and produces the target's NV12
    tensors plus an immutable :class:`LetterboxContext`; ``infer`` executes
    PartA and returns validated float32 outputs; ``postprocess`` runs the
    CPU middle + RoIAlign, executes PartB, and decodes the tail into a
    :class:`DetectionResult` with original COCO labels in original-image
    pixels.  ``predict`` composes the three stages.
    """

    def __init__(self, selection: PairedSelection | None = None, *,
                 runner: PairedRuntime | None = None,
                 runtime_factory: Callable[[Any], Any] | None = None):
        if runner is None:
            if not isinstance(selection, PairedSelection):
                raise TypeError("Pass a PairedSelection from resolve_selection, or inject runner=.")
            runner = PairedRuntime(selection, runtime_factory=runtime_factory).load()
        if not isinstance(runner, PairedRuntime):
            raise TypeError("runner must be a PairedRuntime.")
        self.runner = runner
        self.selection = runner.selection
        self.anchors = generate_anchors()
        self._last_proposals = 0

    # -- stages ----------------------------------------------------------
    def preprocess(self, image: np.ndarray) -> PreparedInput:
        import cv2

        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1):
            raise ValueError("Expected a non-empty HWC uint8 BGR image.")
        context = letterbox_geometry(image.shape[0], image.shape[1])
        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        net_h, net_w = context.net_size
        resized = cv2.resize(image, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
        canvas[:net_h, :net_w] = resized          # BGR stays BGR: the NV12
        # converter consumes BGR and emits bt601_video planes.
        tensors = nv12_tensors(canvas, self.selection.target)
        return PreparedInput(tensors, context)

    def infer(self, prepared: PreparedInput) -> dict[str, np.ndarray]:
        tensors = prepared.tensors if isinstance(prepared, PreparedInput) else prepared
        return self.runner.run_parta(tensors)

    def middle(self, outputs: Mapping[str, np.ndarray]) -> tuple[np.ndarray, int, np.ndarray]:
        """Run the CPU RPN selection; return ``(proposals, n, feat0)``."""
        proposals, n = rpn_select(outputs["rpn_obj"], outputs["rpn_deltas"], self.anchors)
        return proposals, n, outputs["feat0"]

    def pool(self, proposals: np.ndarray, feat0: np.ndarray,
             n_valid: int | None = None) -> np.ndarray:
        """RoIAlign the first ``n_valid`` proposals on ``feat0``.

        ``feat0`` arrives as the PartA output ``[1,256,10,10]`` and is
        reduced to the ``[C,H,W]`` map ``roi_align_feat0`` consumes.  Only
        the valid proposal rows are pooled; padding rows (and rows beyond
        ``n_valid``) stay exactly zero, so the static PartB batch is
        deterministic garbage-free and the tail masks them again on the
        way out.
        """
        feat = np.asarray(feat0, dtype=np.float32)
        if feat.shape == (1, 256, 10, 10):
            feat = feat[0]
        elif feat.shape != (256, 10, 10):
            raise ValueError(f"feat0 must be [1,256,10,10] or [256,10,10], got {feat.shape}.")
        rows = np.asarray(proposals, dtype=np.float32).shape[0]
        n = rows if n_valid is None else int(n_valid)
        if not 0 <= n <= NUM_PROPOSALS:
            raise ValueError(f"n_valid must be within [0,{NUM_PROPOSALS}], got {n_valid}.")
        pooled = np.zeros((NUM_PROPOSALS, 256, POOL_OUTPUT_SIZE, POOL_OUTPUT_SIZE),
                          dtype=np.float32)
        if n:
            pooled[:n] = roi_align_feat0(feat, np.asarray(proposals, dtype=np.float32)[:n])
        return pooled

    def postprocess(self, outputs: Mapping[str, np.ndarray],
                    context: LetterboxContext) -> DetectionResult:
        proposals, n, feat0 = self.middle(outputs)
        self._last_proposals = int(n)
        if n == 0:
            return _empty_result()   # no proposals: skip PartB entirely
        pooled = self.pool(proposals, feat0, n)
        cls, reg = self.runner.run_partb(pooled)
        boxes, scores, labels = tail(cls, reg, proposals, n)
        if boxes.shape[0] == 0:
            return _empty_result()
        return DetectionResult(context.to_original(boxes), scores, labels)

    def predict(self, image: np.ndarray) -> DetectionResult:
        prepared = self.preprocess(image)
        outputs = self.infer(prepared)
        return self.postprocess(outputs, prepared.context)

    @property
    def last_proposals(self) -> int:
        """Proposals surviving the RPN of the last postprocess/predict call
        (0 before the first call); the C++ runtime reports the same
        count."""
        return self._last_proposals


__all__ = [
    "ANCHOR_RATIOS", "ANCHOR_SIZES", "BBOX_XFORM_CLIP", "BOX_DETECTIONS_PER_IMG",
    "BOX_MIN_SIZE", "BOX_NMS_THRESH", "BOX_SCORE_THRESH", "BOX_WEIGHTS", "Detection",
    "DetectionResult", "FasterRCNNTask", "INPUT_SIZE", "LIVE_TENSOR_CONTRACTS",
    "LetterboxContext",
    "NUM_CLASSES", "NUM_PROPOSALS", "PARTB_LAYOUT_PACKED", "PARTB_LAYOUT_POOLED",
    "PairedRuntime", "PairedSelection", "PERSON_LABEL", "POOL_OUTPUT_SIZE",
    "POOL_SAMPLING_RATIO", "POOL_SPATIAL_SCALE", "PreparedInput",
    "RPN_BOX_WEIGHTS", "RPN_MIN_SIZE", "RPN_NMS_THRESH", "RPN_POST_NMS_TOP_N",
    "RPN_PRE_NMS_TOP_N", "RPN_SCORE_THRESH", "batched_nms", "clip_boxes_to_image",
    "decode_deltas", "file_sha256", "generate_anchors", "letterbox_geometry",
    "nms", "nv12_tensors", "pack_bank", "remove_small_boxes",
    "roi_align_feat0", "rpn_select", "tail", "unpack_bank_outputs",
]
