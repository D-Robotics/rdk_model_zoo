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
"""YOLO pose estimation: DFL (v8/11) and direct-LTRB YOLO26 keypoints.

:class:`YoloPose` owns the readable pose stages (preprocess → infer →
postprocess) for the DFL families; :class:`YOLO26Pose` selects the
direct-LTRB COCO-17 contract on the same stages.  Below the classes lives
the numeric pose decode — box/keypoint decoding, visibility sigmoid and
inverse geometry.  The shared image transport and letterbox geometry come
from ``detect.py``; the tensor contracts and runner come from
``backend.py``.
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import numpy as np
from utils.py_utils import (
    postprocess as post,
)

from samples.vision.ultralytics_yolo.runtime.python.backend import (
    DFLPoseContract,
    LTRBPoseContract,
    ModelSelection,
    build_runner,
)
from samples.vision.ultralytics_yolo.runtime.python.cli import PlatformProfile
from samples.vision.ultralytics_yolo.runtime.python.detect import (
    ImageTransform,
    PreparedDetection,
    _forward_runner,
    _normalise_grids,
    _prepare_image,
    _semantic_outputs,
    _set_scheduling_params,
    _size_from_runner,
    _transform_for_postprocess,
    inverse_boxes,
    inverse_points,
    sigmoid,
)

# ====================================================================
# The pose task classes.
# ====================================================================

@dataclass
class YoloPoseConfig:
    """Configuration for initializing the YoloPose model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing, inference, and postprocessing in the YOLO pose
    estimation pipeline. It applies to DFL-based YOLO pose models (v8 and v11).

    Attributes:
        model_path: Path to the compiled YOLO-Pose `.hbm` model.
        resize_type: Image resize mode used during preprocessing.
            - 1: Keep aspect ratio with letterbox padding.
        score_thres: Minimum confidence threshold for filtering detections.
        nms_thres: IoU threshold used for Non-Maximum Suppression.
        reg: Number of DFL regression bins per bounding-box side. Defaults to 16.
        nkpt: Number of keypoints the model predicts. Defaults to 17 (COCO).
        strides: Feature map downsampling strides for each detection scale.
        anchor_sizes: Feature map grid sizes for each detection scale.
    """

    model_path: str
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[Tuple[int, int]] = None
    resize_type: int = 1
    score_thres: float = 0.25
    nms_thres: Optional[float] = None
    reg: int = 16
    nkpt: int = 17
    strides: list = field(default_factory=lambda: [8, 16, 32])
    anchor_sizes: Optional[list] = None

    contract: Optional[DFLPoseContract] = None


class YoloPose:
    """COCO pose stages; subclasses select a reviewed box/keypoint protocol."""

    def __init__(self, config: YoloPoseConfig, runner=None):
        self.cfg = config
        requested = config.contract or DFLPoseContract(
            reg_bins=config.reg, strides=config.strides, nkpt=config.nkpt
        )
        if runner is None:
            runner = build_runner(
                ModelSelection(
                    config.model_path,
                    target=getattr(config.platform, "key", None),
                    platform=config.platform,
                    task="pose",
                    contract=requested,
                    input_shape=config.input_shape,
                )
            )
        self.runner = runner
        self.binding = getattr(runner, "binding", None)
        self.contract = getattr(self.binding, "contract", requested)
        if self.contract.task != "pose":
            raise ValueError("YoloPose requires a pose contract.")
        self.model = getattr(runner, "model", runner)
        self.model_name = getattr(runner, "model_name", None)
        self.input_adapter = getattr(runner, "input_adapter", None)
        self.input_h, self.input_w = _size_from_runner(runner, config)
        self.input_size = (self.input_h, self.input_w)
        grids = _normalise_grids(
            config.anchor_sizes, self.input_size, self.contract.strides
        )
        if any(h != w for h, w in grids):
            raise ValueError("Published pose requires square input grids.")
        self.anchor_sizes = [h for h, w in grids]
        self.nkpt = self.contract.nkpt
        self.input_names = tuple(getattr(runner, "input_names", ()))
        self.output_names = tuple(getattr(runner, "output_names", ()))
        self.input_shapes = dict(getattr(runner, "input_shapes", {}))
        if config.nms_thres is None:
            config.nms_thres = getattr(config.platform, "nms_thres", 0.7)

    def set_scheduling_params(self, priority=None, bpu_cores=None):
        """Forward explicitly supplied scheduling values through the runner."""
        _set_scheduling_params(
            self.runner, self.model, self.model_name, priority, bpu_cores
        )

    def preprocess(self, img, image_format="BGR") -> PreparedDetection:
        """Validate BGR uint8 HxWx3 and prepare NV12 with immutable geometry."""
        tensors, transform = _prepare_image(
            self.runner,
            self.input_adapter,
            self.input_size,
            self.cfg.resize_type,
            img,
            image_format,
        )
        return PreparedDetection(tensors, transform)

    def infer(self, input_tensor):
        """Execute the runner once without changing raw output values or layout."""
        return _forward_runner(self.runner, input_tensor)

    def postprocess(
        self,
        outputs,
        ori_img_w=None,
        ori_img_h=None,
        score_thres=None,
        nms_thres=None,
        transform=None,
    ):
        """Decode matching image geometry and return boxes/scores/IDs/xy/visibility.

        Visibility is sigmoid probability in the maintained Ultralytics protocol.
        """
        context = _transform_for_postprocess(
            transform, ori_img_w, ori_img_h, self.input_size, self.cfg.resize_type
        )
        semantic = _semantic_outputs(
            self.binding,
            self.contract,
            outputs,
            f"{getattr(self.contract, 'protocol', 'DFL')} pose",
        )
        return decode_pose(
            semantic,
            self.contract,
            context,
            self.cfg.score_thres if score_thres is None else score_thres,
            self.cfg.nms_thres if nms_thres is None else nms_thres,
        )

    def predict(self, img, image_format="BGR", score_thres=None, nms_thres=None):
        """Compose all three public stages; keep the same visibility domain."""
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

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin
    # aliases of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, img, image_format="BGR") -> PreparedDetection:
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(img, image_format)

    def forward(self, input_tensor):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(input_tensor)

    def post_process(
        self,
        outputs,
        ori_img_w=None,
        ori_img_h=None,
        score_thres=None,
        nms_thres=None,
        transform=None,
    ):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(
            outputs, ori_img_w, ori_img_h, score_thres, nms_thres, transform)


@dataclass
class YOLO26PoseConfig:
    """Local artifact, target, confidence/NMS thresholds and resize strategy.

    Library NMS default remains 0.65; CLI and legacy adapters explicitly provide
    their platform defaults. Only strides 8/16/32 and COCO-17 are supported.
    """

    model_path: str
    score_thres: float = 0.25
    nms_thres: float = 0.65
    resize_type: int = 1
    strides: list = field(default_factory=lambda: [8, 16, 32])
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[tuple] = None
    contract: Optional[LTRBPoseContract] = None
    anchor_sizes: Optional[list] = None


class YOLO26Pose(YoloPose):
    """Shared raw stages; five owned arrays with one sigmoid visibility."""

    task = "pose"

    def __init__(self, config: YOLO26PoseConfig, runner=None):
        requested = config.contract or LTRBPoseContract(strides=config.strides)
        effective = getattr(getattr(runner, "binding", None), "contract", requested)
        for contract in (requested, effective):
            if (
                contract.task != "pose"
                or getattr(contract, "box_encoding", None) != "ltrb"
                or contract.nkpt != 17
            ):
                raise ValueError(
                    "YOLO26Pose requires the direct-LTRB COCO-17 pose contract."
                )
        super().__init__(replace(config, contract=requested), runner=runner)

# ====================================================================
# Numeric decode: DFL/direct-LTRB pose and visibility; no model loading, SDK or drawing.
# ====================================================================

def decode_pose(outputs, contract, transform, score_thres, nms_thres):
    """Return owned original-image pose arrays after named floating-output binding."""
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
            ("cls", 1),
            ("box", box_channels),
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
    weights = (
        None
        if direct
        else np.arange(contract.reg_bins, dtype=np.float32)[None, None, :]
    )
    boxes, scores, ids, points, logits = [], [], [], [], []
    threshold = -np.log(1.0 / score_thres - 1.0)
    for stride in contract.strides:
        conf, classes, selected = post.filter_classification(
            outputs[f"cls_{stride}"], threshold
        )
        if direct:
            anchors = post.gen_anchor(height // stride)[selected]
            offsets = outputs[f"box_{stride}"].reshape(-1, 4)[selected]
            boxes.append(post.decode_ltrb_boxes(anchors, offsets, stride))
            keypoints = outputs[f"kpts_{stride}"].reshape(-1, contract.nkpt, 3)[
                selected
            ]
            xy = (keypoints[..., :2] + anchors[:, None, :]) * stride
            visibility = keypoints[..., 2:3]
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

__all__ = ["YOLO26Pose", "YOLO26PoseConfig", "YoloPose", "YoloPoseConfig",
           "decode_pose"]
