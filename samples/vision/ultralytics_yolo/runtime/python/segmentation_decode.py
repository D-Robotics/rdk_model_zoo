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

"""DFL/direct-LTRB instance mask decoding; no model loading, SDK calls or rendering."""

import cv2
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.decode import sigmoid
from utils.py_utils import (
    postprocess as post,
)
from samples.vision.ultralytics_yolo.runtime.python.geometry import inverse_boxes


def decode_segmentation(
    outputs, contract, transform, score_thres, nms_thres, *, do_morph=True
):
    """Decode validated NHWC floating semantic heads and stride-4 prototypes.

    Returns owned float32 boxes/scores, int64 IDs and ROI masks (DFL uint8, LTRB bool). Source
    coefficient/prototype math, classwise NMS and optional opening are preserved.
    """
    score_thres, nms_thres = float(score_thres), float(nms_thres)
    if not np.isfinite(score_thres) or not 0 < score_thres < 1:
        raise ValueError("score_thres must be finite and strictly between 0 and 1.")
    if not np.isfinite(nms_thres) or not 0 <= nms_thres <= 1:
        raise ValueError("nms_thres must be finite and between 0 and 1.")
    height, width = transform.model_size
    direct = getattr(contract, "box_encoding", None) == "ltrb"
    box_channels = 4 if direct else 4 * contract.reg_bins
    required = {
        f"{kind}_{stride}": (1, height // stride, width // stride, channels)
        for stride in contract.strides
        for kind, channels in [
            ("cls", contract.classes),
            ("box", box_channels),
            ("mces", contract.mces_num),
        ]
    }
    required["protos"] = (1, height // 4, width // 4, contract.mces_num)
    if set(outputs) != set(required):
        raise ValueError(
            "Segmentation requires exactly the declared semantic output roles."
        )
    for name, shape in required.items():
        value = np.asarray(outputs[name])
        if (
            value.shape != shape
            or value.dtype.kind != "f"
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                f"Semantic output {name!r} must be finite floating NHWC {shape}."
            )
    boxes, scores, ids, coefficients = [], [], [], []
    weights = (
        None
        if direct
        else np.arange(contract.reg_bins, dtype=np.float32)[None, None, :]
    )
    threshold = -np.log(1.0 / score_thres - 1.0)
    for stride in contract.strides:
        conf, classes, selected = post.filter_classification(
            outputs[f"cls_{stride}"], threshold
        )
        if direct:
            anchors = post.gen_anchor(height // stride)[selected]
            offsets = outputs[f"box_{stride}"].reshape(-1, 4)[selected]
            boxes.append(post.decode_ltrb_boxes(anchors, offsets, stride))
        else:
            boxes.append(
                post.decode_boxes(
                    outputs[f"box_{stride}"],
                    selected,
                    height // stride,
                    stride,
                    weights,
                )
            )
        scores.append(conf)
        ids.append(classes)
        coefficients.append(post.filter_mces(outputs[f"mces_{stride}"], selected))
    boxes, scores, ids, coefficients = [
        np.concatenate(parts, axis=0) for parts in (boxes, scores, ids, coefficients)
    ]
    keep = post.NMS(boxes, scores, ids, nms_thres)
    if direct:
        xyxy = inverse_boxes(boxes[keep], transform)
        masks = _probability_roi_masks(
            outputs["protos"][0], coefficients[keep], boxes[keep], xyxy, transform
        )
        return (
            np.array(xyxy, dtype=np.float32, copy=True),
            np.array(scores[keep], dtype=np.float32, copy=True),
            np.array(ids[keep], dtype=np.int64, copy=True),
            masks,
        )
    proto = outputs["protos"][0]
    # Source semantics: the prototype crop uses the full decoded box, letterbox
    # padding included (the model produces prototype responses there too).
    # Boxes are not clipped to the content region, and negative crop starts
    # keep the source NumPy slicing behavior, so masks match the original
    # sample output exactly.
    masks = post.decode_masks(
        coefficients[keep],
        boxes[keep],
        proto,
        width,
        height,
        proto.shape[1],
        proto.shape[0],
        mask_thresh=0.5,
    )
    xyxy = inverse_boxes(boxes[keep], transform)
    oh, ow = transform.original_size
    masks = post.resize_masks_to_boxes(masks, xyxy, ow, oh, do_morph=do_morph)
    return (
        np.array(xyxy, dtype=np.float32, copy=True),
        np.array(scores[keep], dtype=np.float32, copy=True),
        np.array(ids[keep], dtype=np.int64, copy=True),
        [np.array(mask, dtype=np.uint8, copy=True) for mask in masks],
    )


def _probability_roi_masks(
    prototype, coefficients, model_boxes, original_boxes, transform
):
    """YOLO26: interpolate probabilities, remove actual padding, then binarize.

    This order intentionally differs from DFL's cropped binary-mask resizing.
    No morphological opening is applied. Return one owned bool ROI per box.
    """
    if len(coefficients) == 0:
        return []
    mh, mw, channels = prototype.shape
    logits = coefficients @ prototype.reshape(-1, channels).T
    probabilities = sigmoid(logits).reshape(-1, mh, mw)
    ih, iw = transform.model_size
    oh, ow = transform.original_size
    left, top, right, bottom = transform.padding
    rows = np.arange(ih, dtype=np.float32)[:, None]
    columns = np.arange(iw, dtype=np.float32)[None, :]
    result = []
    for probability, box, original in zip(probabilities, model_boxes, original_boxes):
        probability = cv2.resize(probability, (iw, ih), interpolation=cv2.INTER_LINEAR)
        probability *= (
            (columns >= box[0])
            & (columns < box[2])
            & (rows >= box[1])
            & (rows < box[3])
        )
        content = probability[top : ih - bottom, left : iw - right]
        restored = cv2.resize(content, (ow, oh), interpolation=cv2.INTER_LINEAR)
        x1, y1, x2, y2 = np.asarray(original, dtype=np.int64)
        if x2 > x1 and y2 > y1:
            result.append(np.array(restored[y1:y2, x1:x2] > 0.5, dtype=bool, copy=True))
        else:
            result.append(np.zeros((0, 0), dtype=bool))
    return result
