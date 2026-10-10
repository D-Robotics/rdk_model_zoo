# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLOE prompt-free segmentation task: prepare, one raw call, variant decode.

:class:`YOLOE` owns the real per-call flow in this one file:
:meth:`preprocess` turns one BGR image into the bound NV12 tensors plus
this image's geometry context, :meth:`infer` executes exactly one runner
call, and :meth:`postprocess` decodes the semantic outputs for the selected
variant — YOLOE-26 (direct candidates plus ROI masks), X5 YOLOE-11 (source
DFL/NMS with full-image masks) and S YOLOE-11 (ROI segmentation with
optional morphology). The shared decode entry :func:`decode_result` is what
the offline float evaluator reuses; the fixed thresholds and geometry
restoration come from the shared helpers in ``utils/py_utils`` and the
Ultralytics geometry/segmentation modules. ``Config`` is re-exported from
``config.py``, which the native launcher also imports.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from utils.py_utils import postprocess as post
from utils.py_utils.image import bgr_to_nv12_planes
from utils.py_utils.yoloe26_decode import decode_candidates, restore_masks
from utils.py_utils.yoloe26_geometry import PFGeometry, letterbox
from samples.vision.ultralytics_yolo.runtime.python.detection_io import (
    _semantic_outputs,
)
from samples.vision.ultralytics_yolo.runtime.python.geometry import (
    ImageTransform,
    inverse_boxes,
    resize_with_transform,
)
from samples.vision.ultralytics_yolo.runtime.python.segmentation_decode import (
    decode_segmentation,
)
from samples.vision.yoloe.runtime.python.config import Config, validate_config
from samples.vision.yoloe.runtime.python.model_binding import runtime_selection


@dataclass(frozen=True)
class Prepared:
    """Owned NV12 tensors plus this image's immutable geometry context."""

    tensors: Any
    context: Any


@dataclass(frozen=True)
class Result:
    """Owned detection arrays plus the declared mask layout.

    boxes: float32 (N,4) in original-image pixels. scores: float32 (N,).
    class_ids: int64 (N,). masks: bool (N,H,W) for ``full`` layout or the
    per-detection ROI objects for ``roi`` layout; ``mask_layout`` names
    which. Arrays are owned by the result.
    """

    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    masks: Any
    mask_layout: str


def validate_context(context, selection, cfg):
    """Reject a context that does not belong to this variant and config.

    Args:
        context: The geometry object carried by :class:`Prepared`.
        selection: Resolved YOLOE selection naming the variant.
        cfg: Config naming the expected resize policy.

    Raises:
        ValueError: When the context type or geometry disagrees.
    """
    if selection.variant.startswith("26"):
        if not isinstance(context, PFGeometry):
            raise ValueError("YOLOE-26 requires its prepared PFGeometry.")
    elif (
        not isinstance(context, ImageTransform)
        or context.model_size != (640, 640)
        or context.resize_type != cfg.resize_type
    ):
        raise ValueError("YOLOE-11 requires its matching prepared ImageTransform.")


class YOLOE:
    """Prompt-free segmentation with no file IO or last-image state."""

    def __init__(self, selection, config=None, *, runner=None):
        self.selection = selection
        self.cfg = config or Config()
        validate_config(selection, self.cfg)
        selected = runtime_selection(selection)
        if runner is None:
            # Resolve through the module so host tests can inject an SDK
            # factory by patching model_runner.build_runner.
            from samples.vision.yoloe.runtime.python import model_runner

            runner = model_runner.build_runner(selection)
        self.runner = runner
        self.binding = self.runner.binding
        if self.binding.selection != selected or self.runner.input_size != (640, 640):
            raise ValueError("Injected runner does not match the YOLOE selection.")
        self.contract = self.binding.contract

    def set_scheduling_params(self, *, priority=None, bpu_cores=None):
        """Apply explicit scheduling values to the loaded board runtime."""
        self.runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

    def preprocess(self, image):
        """BGR uint8 HWC -> NV12 tensors plus immutable per-image context.

        YOLOE-26 uses the fixed round/114 letterbox; YOLOE-11 uses the
        configured stretch/letterbox transform. Raises ValueError for an
        image that is not a nonempty uint8 HWC BGR array.
        """
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or min(image.shape[:2]) <= 0
        ):
            raise ValueError("Expected a nonempty uint8 HWC BGR image.")
        if self.selection.variant.startswith("26"):
            pixels, context = letterbox(image)
        else:
            pixels, context = resize_with_transform(image, (640, 640), self.cfg.resize_type)
        y, uv = bgr_to_nv12_planes(pixels)
        return Prepared(self.runner.prepare_input(y, uv), context)

    def infer(self, prepared):
        """Exactly one runner call; return borrowed native arrays unchanged."""
        return self.runner(
            prepared.tensors if isinstance(prepared, Prepared) else prepared
        )

    def postprocess(self, outputs, context):
        """Raw outputs plus matching context -> owned original-coordinate results."""
        semantic = _semantic_outputs(self.binding, self.contract, outputs, "YOLOE PF")
        return decode_result(semantic, self.contract, self.selection, self.cfg, context)

    def predict(self, image):
        """Compose the public stages, carrying this image's context explicitly."""
        prepared = self.preprocess(image)
        return self.postprocess(self.infer(prepared), prepared.context)

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin
    # aliases of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, image):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image)

    def forward(self, prepared):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(prepared)

    def post_process(self, outputs, context):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs, context)


def decode_result(semantic, contract, selection, cfg, context):
    """Decode one semantic output mapping for the selected variant.

    Shared by :meth:`YOLOE.postprocess` and the offline float evaluator.

    Args:
        semantic: Mapping of the ten semantic output roles to float arrays.
        contract: Bound variant contract (channel counts, required roles).
        selection: Resolved YOLOE selection naming target and variant.
        cfg: Config with score/NMS/morphology/max-det policy.
        context: The geometry context carried by this call's Prepared.

    Returns:
        Result: Owned boxes/scores/class_ids plus the variant's masks and
        the declared mask layout (``roi`` or ``full``).

    Raises:
        ValueError: On a config/context/semantic contract violation.
    """
    validate_config(selection, cfg)
    validate_context(context, selection, cfg)
    semantic = validate_semantic(semantic, contract)
    if selection.variant.startswith("26"):
        ordered = [semantic[role] for role in contract.required_roles]
        boxes, scores, ids, coefficients = decode_candidates(
            ordered, cfg.score_thres, cfg.max_det, cfg.single_label
        )
        boxes, masks = restore_masks(
            boxes, coefficients, semantic["protos"][0], context
        )
        return Result(boxes, scores, ids, masks, "roi")
    nms = 0.7 if cfg.nms_thres is None else cfg.nms_thres
    if selection.target == "x5":
        values = decode_x5(semantic, contract, context, cfg.score_thres, nms)
        return Result(*values, "full")
    values = decode_segmentation(
        semantic, contract, context, cfg.score_thres, nms, do_morph=cfg.do_morph
    )
    return Result(*values, "roi")


def decode_x5(outputs, contract, context, score_thres, nms_thres):
    """Keep source DFL/NMS, prototype probability crop and full-image bool masks.

    Args:
        outputs: Semantic role mapping with float32 cls/box/mces/protos heads.
        contract: Bound YOLOE-11 contract supplying the box channel count.
        context: ImageTransform of this call; supplies padding and sizes.
        score_thres: Score gate applied to objectness*class scores.
        nms_thres: Class-agnostic NMS IoU threshold.

    Returns:
        Tuple of boxes (N,4) float32, scores (N,) float32, class ids (N,)
        int64, and bool masks (N,H,W) at the original image size.
    """
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
    """An injected semantic mapping must satisfy the same float contract as the SDK.

    Args:
        outputs: Candidate semantic role mapping.
        contract: Bound contract supplying the box channel count.

    Returns:
        The same mapping, unchanged, after validation.

    Raises:
        ValueError: On missing/extra roles or wrong shape/dtype/finiteness.
    """
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
