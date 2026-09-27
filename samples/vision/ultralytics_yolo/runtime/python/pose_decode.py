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

"""DFL pose decoding and visibility; no model loading, SDK or drawing."""

import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.rdk_yolo_utils import (
    postprocess as post,
)
from samples.vision.ultralytics_yolo.runtime.python.geometry import (
    inverse_boxes,
    inverse_points,
)
from samples.vision.ultralytics_yolo.runtime.python.decode import sigmoid


def decode_pose(outputs, contract, transform, score_thres, nms_thres):
    """Return owned original-image pose arrays after named floating-output binding."""
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
            ("cls", 1),
            ("box", 4 * contract.reg_bins),
            ("kpts", 3 * contract.nkpt),
        ]
    }
    if set(outputs) != set(required):
        raise ValueError("Pose requires exactly the declared semantic output roles.")
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
    weights = np.arange(contract.reg_bins, dtype=np.float32)[None, None, :]
    boxes, scores, ids, points, logits = [], [], [], [], []
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
        xy, visibility = post.decode_kpts(
            outputs[f"kpts_{stride}"], selected, height // stride, stride
        )
        scores.append(conf)
        ids.append(classes)
        points.append(xy)
        logits.append(visibility)
    boxes, scores, ids, points, logits = [
        np.concatenate(values, axis=0)
        for values in (boxes, scores, ids, points, logits)
    ]
    keep = post.NMS(boxes, scores, ids, nms_thres)
    xyxy = inverse_boxes(boxes[keep], transform)
    points = inverse_points(points[keep], transform)
    visibility = sigmoid(logits[keep])
    return (
        np.array(xyxy, dtype=np.float32, copy=True),
        np.array(scores[keep], dtype=np.float32, copy=True),
        np.array(ids[keep], dtype=np.int64, copy=True),
        points,
        np.array(visibility, dtype=np.float32, copy=True),
    )
