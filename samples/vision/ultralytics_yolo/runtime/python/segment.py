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
"""YOLO instance segmentation: DFL (v8/9/11) and direct-LTRB YOLO26.

:class:`YoloSeg` owns the readable segmentation stages (preprocess → infer →
postprocess) for the DFL families; :class:`YOLO26Seg` selects the
direct-LTRB, 32-coefficient contract on the same stages.  Below the classes
lives the numeric segmentation decode — source coefficient/prototype mask
math, classwise NMS, YOLO26's probability-ROI binarisation.  The shared
image transport and letterbox geometry come from ``detect.py``; the tensor
contracts and runner come from ``backend.py``.
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import cv2
import numpy as np
from utils.py_utils import (
    postprocess as post,
)

from samples.vision.ultralytics_yolo.runtime.python.backend import (
    DFLSegmentationContract,
    LTRBSegmentationContract,
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
    sigmoid,
)

# ====================================================================
# The segmentation task classes.
# ====================================================================

@dataclass
class YoloSegConfig:
    """Configuration for initializing the YoloSeg model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing, inference, and postprocessing in the YOLO segmentation
    pipeline. It applies to DFL-based YOLO segmentation models (v8, v9 and v11).

    Attributes:
        model_path: Path to the compiled YOLO-Seg `.hbm` model.
        classes_num: Number of detection classes. Defaults to 80 (COCO).
        resize_type: Image resize mode used during preprocessing.
            - 1: Keep aspect ratio with letterbox padding.
        score_thres: Minimum confidence threshold for filtering detections.
        nms_thres: IoU threshold used for Non-Maximum Suppression.
        reg: Number of DFL regression bins per bounding-box side. Defaults to 16.
        mces_num: Dimension of the MCES (mask coefficient) vector. Defaults to 32.
        strides: Feature map downsampling strides for each detection scale.
        anchor_sizes: Feature map grid sizes for each detection scale.
        do_morph: Whether to apply morphological opening to clean mask edges.
    """

    model_path: str
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[Tuple[int, int]] = None
    classes_num: int = 80
    resize_type: int = 1
    score_thres: float = 0.25
    nms_thres: Optional[float] = None
    reg: int = 16
    mces_num: int = 32
    strides: list = field(default_factory=lambda: [8, 16, 32])
    anchor_sizes: Optional[list] = None
    do_morph: bool = True

    contract: Optional[DFLSegmentationContract] = None


class YoloSeg:
    """Shared segmentation stages with a replaceable raw model runner.

    Loading/metadata and scheduling belong to the runner. Image geometry belongs
    to each PreparedDetection, never a cached last image. Physical output order
    is not a model contract. See runtime/python/README for tuple mask semantics.
    """

    def __init__(self, config: YoloSegConfig, runner=None):
        self.cfg = config
        requested = config.contract or DFLSegmentationContract(
            classes=config.classes_num,
            reg_bins=config.reg,
            strides=config.strides,
            mces_num=config.mces_num,
        )
        if runner is None:
            runner = build_runner(
                ModelSelection(
                    config.model_path,
                    target=getattr(config.platform, "key", None),
                    platform=config.platform,
                    task="segment",
                    contract=requested,
                    input_shape=config.input_shape,
                )
            )
        self.runner = runner
        self.binding = getattr(runner, "binding", None)
        self.contract = getattr(self.binding, "contract", requested)
        if self.contract.task != "segment":
            raise ValueError("YoloSeg requires a segmentation contract.")
        self.model = getattr(runner, "model", runner)
        self.model_name = getattr(runner, "model_name", None)
        self.input_adapter = getattr(runner, "input_adapter", None)
        self.input_h, self.input_w = _size_from_runner(runner, config)
        self.input_size = (self.input_h, self.input_w)
        grids = _normalise_grids(
            config.anchor_sizes, self.input_size, self.contract.strides
        )
        if any(h != w for h, w in grids):
            raise ValueError("Published segmentation requires square input grids.")
        self.anchor_sizes = [h for h, w in grids]
        self.input_names = tuple(getattr(runner, "input_names", ()))
        self.output_names = tuple(getattr(runner, "output_names", ()))
        self.input_shapes = dict(getattr(runner, "input_shapes", {}))
        if config.nms_thres is None:
            config.nms_thres = getattr(config.platform, "nms_thres", 0.7)

    def set_scheduling_params(self, priority=None, bpu_cores=None):
        """Forward only explicitly supplied scheduler settings."""
        _set_scheduling_params(
            self.runner, self.model, self.model_name, priority, bpu_cores
        )

    def preprocess(self, img, image_format="BGR") -> PreparedDetection:
        """Return owned NV12 tensors and immutable geometry for BGR uint8 HxWx3."""
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
        """Call the runner once, preserving raw floating dtype and layout."""
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
        """Decode/NMS and return original-image boxes and ROI masks.

        Pass the matching prepared.transform (preferred) or both original image
        dimensions (legacy). Results own their storage and survive SDK reuse.
        """
        context = _transform_for_postprocess(
            transform, ori_img_w, ori_img_h, self.input_size, self.cfg.resize_type
        )
        semantic = _semantic_outputs(
            self.binding,
            self.contract,
            outputs,
            f"{getattr(self.contract, 'protocol', 'DFL')} segmentation",
        )
        return decode_segmentation(
            semantic,
            self.contract,
            context,
            self.cfg.score_thres if score_thres is None else score_thres,
            self.cfg.nms_thres if nms_thres is None else nms_thres,
            do_morph=self.cfg.do_morph,
        )

    def predict(self, img, image_format="BGR", score_thres=None, nms_thres=None):
        """Compose the same three public stages without I/O or last-image state."""
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
class YOLO26SegConfig:
    """Local artifact/target and thresholds; 32 coefficients, no mask morphology.

    Library NMS default remains 0.65; CLI supplies its platform default.
    resize_type is 0=stretch or 1=letterbox.
    """

    model_path: str
    classes_num: int = 80
    score_thres: float = 0.25
    nms_thres: float = 0.65
    resize_type: int = 1
    strides: list = field(default_factory=lambda: [8, 16, 32])
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[tuple] = None
    contract: Optional[LTRBSegmentationContract] = None
    anchor_sizes: Optional[list] = None
    do_morph: bool = field(default=False, init=False)


class YOLO26Seg(YoloSeg):
    """Shared raw stages returning owned boxes/scores/IDs and boolean ROI masks."""

    task = "seg"

    def __init__(self, config: YOLO26SegConfig, runner=None):
        requested = config.contract or LTRBSegmentationContract(
            classes=config.classes_num, strides=config.strides
        )
        effective = getattr(getattr(runner, "binding", None), "contract", requested)
        for contract in (requested, effective):
            if (
                contract.task != "segment"
                or getattr(contract, "box_encoding", None) != "ltrb"
                or contract.mces_num != 32
            ):
                raise ValueError(
                    "YOLO26Seg requires the direct-LTRB 32-coefficient segmentation contract."
                )
        super().__init__(replace(config, contract=requested), runner=runner)

# ====================================================================
# Numeric decode: DFL/direct-LTRB instance masks; no model loading, SDK calls or rendering.
# ====================================================================

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

__all__ = ["YOLO26Seg", "YOLO26SegConfig", "YoloSeg", "YoloSegConfig",
           "decode_segmentation"]
