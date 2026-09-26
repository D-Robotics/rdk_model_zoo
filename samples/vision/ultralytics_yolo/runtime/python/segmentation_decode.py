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

"""DFL instance mask decoding; no model loading, SDK calls or rendering."""

import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.rdk_yolo_utils import (
    postprocess as post,
)
from samples.vision.ultralytics_yolo.runtime.python.geometry import inverse_boxes


def decode_segmentation(
    outputs, contract, transform, score_thres, nms_thres, *, do_morph=True
):
    """Decode validated NHWC floating semantic heads and stride-4 prototypes.

    Returns owned float32 boxes/scores, int64 IDs and uint8 ROI masks. Source
    coefficient/prototype math, classwise NMS and optional opening are preserved.
    """
    score_thres, nms_thres = float(score_thres), float(nms_thres)
    if not np.isfinite(score_thres) or not 0 < score_thres < 1:
        raise ValueError("score_thres must be finite and strictly between 0 and 1.")
    if not np.isfinite(nms_thres) or not 0 <= nms_thres <= 1:
        raise ValueError("nms_thres must be finite and between 0 and 1.")
    height, width = transform.model_size
    required = {
        f"{kind}_{stride}": (1, height // stride, width // stride, channels)
        for stride in contract.strides
        for kind, channels in [
            ("cls", contract.classes),
            ("box", 4 * contract.reg_bins),
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
    weights = np.arange(contract.reg_bins, dtype=np.float32)[None, None, :]
    threshold = -np.log(1.0 / score_thres - 1.0)
    for stride in contract.strides:
        conf, classes, selected = post.filter_classification(
            outputs[f"cls_{stride}"], threshold
        )
        boxes.append(
            post.decode_boxes(
                outputs[f"box_{stride}"], selected, height // stride, stride, weights
            )
        )
        scores.append(conf)
        ids.append(classes)
        coefficients.append(post.filter_mces(outputs[f"mces_{stride}"], selected))
    boxes, scores, ids, coefficients = [
        np.concatenate(parts, axis=0) for parts in (boxes, scores, ids, coefficients)
    ]
    keep = post.NMS(boxes, scores, ids, nms_thres)
    proto = outputs["protos"][0]
    # Clip the prototype crop to image content, excluding letterbox padding.
    # Negative NumPy starts otherwise address the opposite edge of the tensor.
    visible_boxes = boxes[keep].copy()
    left, top, right, bottom = transform.padding
    visible_boxes[:, (0, 2)] = np.clip(visible_boxes[:, (0, 2)], left, width - right)
    visible_boxes[:, (1, 3)] = np.clip(visible_boxes[:, (1, 3)], top, height - bottom)
    masks = post.decode_masks(
        coefficients[keep],
        visible_boxes,
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
