# Copyright (c) 2025-2026 D-Robotics Corporation
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
"""YOLO26 rotated-box (OBB) detection.

:class:`YOLO26OBB` owns the readable rotated-box stages; it borrows the
protocol-independent image transport and raw model call from
:class:`detect.YoloDetect` and adds the OBB-specific postprocess.  Below the
class lives the numeric rotated decode — radians, the source rotated-NMS
policies (classwise X5 vs ``cv2.dnn.NMSBoxesRotated`` on S) and the inverse
geometry.  The tensor contract and runner come from ``backend.py``.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from samples.vision.ultralytics_yolo.runtime.python.backend import (
    LTRBOBBContract,
    ModelSelection,
    build_runner,
)
from samples.vision.ultralytics_yolo.runtime.python.cli import (
    PlatformProfile,
    resolve_platform,
)
from samples.vision.ultralytics_yolo.runtime.python.detect import (
    YoloDetect,
    _normalise_grids,
    _semantic_outputs,
    _size_from_runner,
    _transform_for_postprocess,
    sigmoid,
)

# ====================================================================
# The rotated-box task class.
# ====================================================================

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

# ====================================================================
# Numeric decode: pure rotated-box decoding, source NMS policies and explicit inverse geometry.
# ====================================================================

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

__all__ = ["YOLO26OBB", "YOLO26OBBConfig", "decode_obb"]
