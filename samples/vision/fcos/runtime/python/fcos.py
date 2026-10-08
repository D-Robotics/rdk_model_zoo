# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""FCOS readable task: preprocess, infer, postprocess, and predict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from utils.py_utils.image import bgr_to_nv12_planes
from utils.py_utils.quantization import apply_output_transform


@dataclass(frozen=True)
class ImageContext:
    """Geometry for one call; no mutable task state is used."""

    original_shape: tuple[int, int]
    input_shape: tuple[int, int]
    resize_type: int
    resized_shape: tuple[int, int]
    pad: tuple[int, int, int, int]


@dataclass(frozen=True)
class PreparedInput:
    """Packed tensors plus the context consumed by FCOS post-processing."""

    tensors: dict[str, np.ndarray]
    context: ImageContext


def prepare(image: np.ndarray, binding, *, resize_type: int | None = None) -> PreparedInput:
    """Convert BGR uint8 input to one flat packed NV12 tensor.

    Args:
        image: Nonempty uint8 BGR array shaped (H, W, 3); not modified.
        binding: ModelBinding supplying input names, geometry, and validation.
        resize_type: Optional 0 (direct) or 1 (letterbox) override; None uses
            the binding's contract default.

    Returns:
        PreparedInput: One flat contiguous uint8 packed-NV12 tensor keyed by
        the bound input name, plus this call's frozen geometry context.

    Raises:
        ValueError: For invalid pixels, an unsupported resize mode, or a
            degenerate letterbox geometry.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("FCOS input must be a BGR array with shape (H,W,3).")
    if image.dtype != np.uint8 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("FCOS input must be a non-empty uint8 BGR array.")
    chosen = binding.contract.resize_type if resize_type is None else resize_type
    if chosen not in (0, 1):
        raise ValueError("resize_type must be 0 (direct) or 1 (letterbox).")
    height, width = image.shape[:2]
    target_h, target_w = binding.input_height, binding.input_width
    if chosen == 0:
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        pad = (0, 0, 0, 0)
        resized_shape = (target_h, target_w)
    else:
        scale = min(target_h / height, target_w / width)
        new_w, new_h = int(width * scale), int(height * scale)
        resized_small = cv2.resize(image, (new_w, new_h))
        pad_w, pad_h = target_w - new_w, target_h - new_h
        left, right = pad_w // 2, pad_w - pad_w // 2
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        resized = cv2.copyMakeBorder(resized_small, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        pad = (top, bottom, left, right)
        resized_shape = (new_h, new_w)
    y, uv = bgr_to_nv12_planes(resized)
    packed = np.ascontiguousarray(np.concatenate((y.reshape(-1), uv.reshape(-1))), dtype=np.uint8)
    tensors = {binding.input_names[0]: packed}
    binding.validate_inputs(tensors)
    return PreparedInput(tensors=tensors, context=ImageContext((height, width), (target_h, target_w), chosen, resized_shape, pad))


@dataclass(frozen=True)
class DetectionResult:
    """Owned FCOS detections in original-image ``xyxy`` pixel coordinates."""

    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the three legacy-compatible detection arrays."""
        return self.boxes, self.scores, self.class_ids

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DetectionResult):
            return NotImplemented
        return np.array_equal(self.boxes, other.boxes) and np.array_equal(self.scores, other.scores) and np.array_equal(self.class_ids, other.class_ids)


class FCOSTask:
    """Run one bound X5 FCOS model without hidden mutable image context.

    Input is a non-empty BGR ``uint8`` ``(H,W,3)`` array.  Tensors are one
    contiguous packed-NV12 ``uint8`` vector of ``input_h*input_w*3/2`` bytes.
    Context is an immutable :class:`ImageContext`.  RawOutputs are the exact
    fifteen runtime arrays after shape/dtype validation; quantization is owned
    by ``postprocess``.  Result contains owned float32 ``boxes (N,4)`` in
    original-image ``xyxy`` pixels, float32 ``scores (N,)``, and int32
    ``class_ids (N,)``.  ``ValueError`` identifies invalid input, metadata, or
    quantization contracts before model math is attempted.  ``predict``
    composes ``preprocess`` → ``infer`` → ``postprocess``; the established
    ``pre_process``/``forward``/``post_process`` names stay thin aliases.
    """

    def __init__(self, selection=None, *, runner=None, binding=None, runtime_factory=None,
                 conf_thres: float | None = None, iou_thres: float | None = None,
                 resize_type: int | None = None):
        if runner is None:
            from samples.vision.fcos.runtime.python.model_binding import ModelSelection
            from samples.vision.fcos.runtime.python.model_runner import RuntimeModelRunner
            if not isinstance(selection, ModelSelection):
                raise TypeError("Pass a ModelSelection from resolve_selection, or inject runner=/binding=.")
            runner = RuntimeModelRunner(selection, runtime_factory=runtime_factory)
            binding = runner.load()
        elif binding is None:
            binding = getattr(runner, "binding", None)
            if binding is None:
                raise ValueError("An injected runner must supply or be given its binding.")
        if not callable(runner):
            raise TypeError("runner must be callable.")
        self.runner = runner
        self.binding = binding
        self.conf_thres = binding.contract.conf_thres if conf_thres is None else float(conf_thres)
        self.iou_thres = binding.contract.iou_thres if iou_thres is None else float(iou_thres)
        self.resize_type = resize_type
        if not 0 <= self.conf_thres <= 1 or not 0 <= self.iou_thres <= 1:
            raise ValueError("conf_thres and iou_thres must be in [0,1].")

    def set_scheduling_params(self, *, priority: int | None = None,
                              bpu_cores: list[int] | None = None) -> None:
        """Apply explicit scheduling values to the loaded board runtime."""
        self.runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

    # ------------------------------------------------------------------
    # The three pipeline stages, each public and usable on its own.
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> PreparedInput:
        """Produce packed NV12 tensors and a frozen per-call geometry context."""
        return prepare(image, self.binding, resize_type=self.resize_type)

    def infer(self, prepared: PreparedInput | Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Invoke the runner and validate raw tensors without decoding them."""
        tensors = prepared.tensors if isinstance(prepared, PreparedInput) else prepared
        outputs = self.runner(tensors)
        if isinstance(outputs, Mapping) and self.binding.model_name in outputs and isinstance(outputs[self.binding.model_name], Mapping):
            outputs = outputs[self.binding.model_name]
        if not isinstance(outputs, Mapping):
            raise ValueError("FCOS runner must return a mapping of raw output tensors.")
        return self.binding.validate_outputs(outputs)

    def postprocess(self, outputs: Mapping[str, np.ndarray], context: ImageContext) -> DetectionResult:
        """Dequantize, decode FCOS heads, apply source NMS, and restore pixels."""
        _validate_context(context, self.binding)
        raw = self.binding.validate_outputs(outputs)
        transformed = apply_output_transform("dequant", raw, self.binding.output_quants)
        stride_values = self.binding.contract.strides
        grids = _grids(self.binding.input_height, stride_values)
        scores_all: list[np.ndarray] = []
        ids_all: list[np.ndarray] = []
        boxes_all: list[np.ndarray] = []
        for stride, cls_name, center_name, box_name in zip(stride_values, self.binding.cls_output_names, self.binding.center_output_names, self.binding.box_output_names):
            cls = np.asarray(transformed[cls_name], dtype=np.float32).reshape(-1, self.binding.contract.classes_num)
            center = np.asarray(transformed[center_name], dtype=np.float32).reshape(-1, 1)
            raw_max = np.max(cls, axis=1)
            confidence = np.sqrt(_sigmoid(center[:, 0]) * _sigmoid(raw_max))
            valid = np.flatnonzero(confidence >= self.conf_thres)
            scores_all.append(confidence[valid].astype(np.float32, copy=True))
            ids_all.append(np.argmax(cls[valid], axis=1).astype(np.int32, copy=True))
            grid = grids[stride][valid]
            box = np.asarray(transformed[box_name], dtype=np.float32).reshape(-1, 4)[valid]
            box = box * stride
            boxes_all.append(np.hstack((grid - box[:, :2], grid + box[:, 2:4])).astype(np.float32, copy=True))
        if not boxes_all or not any(len(item) for item in boxes_all):
            return _empty_result()
        xyxy = np.concatenate(boxes_all, axis=0).astype(np.float32, copy=True)
        scores = np.concatenate(scores_all, axis=0).astype(np.float32, copy=True)
        class_ids = np.concatenate(ids_all, axis=0).astype(np.int32, copy=True)
        xywh = xyxy.copy()
        xywh[:, 2:] -= xywh[:, :2]
        nms = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), self.conf_thres, self.iou_thres)
        if len(nms) == 0:
            return _empty_result()
        keep = np.asarray(nms).reshape(-1)
        boxes = xyxy[keep].copy()
        if context.resize_type == 0:
            # This is the source FCOS behavior: direct resize maps model
            # coordinates by independent image ratios.
            boxes[:, [0, 2]] *= context.original_shape[1] / self.binding.input_width
            boxes[:, [1, 3]] *= context.original_shape[0] / self.binding.input_height
        elif context.resize_type == 1:
            # Letterbox uses the effective integer resize dimensions recorded
            # by prepare(); using those dimensions preserves the exact
            # cv2.resize rounding instead of reconstructing a nominal scale.
            resized_h, resized_w = context.resized_shape
            if resized_h <= 0 or resized_w <= 0:
                raise ValueError(f"Invalid letterbox resized shape {context.resized_shape!r}.")
            top, _bottom, left, _right = context.pad
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) * context.original_shape[1] / resized_w
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) * context.original_shape[0] / resized_h
        else:
            raise ValueError(f"Unsupported resize_type in ImageContext: {context.resize_type!r}.")
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, context.original_shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, context.original_shape[0])
        return DetectionResult(boxes.astype(np.float32, copy=True), scores[keep].astype(np.float32, copy=True), class_ids[keep].astype(np.int32, copy=True))

    def predict(self, image: np.ndarray) -> DetectionResult:
        """Run exactly ``preprocess → infer → postprocess`` for one image."""
        prepared = self.preprocess(image)
        return self.postprocess(self.infer(prepared), prepared.context)

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin aliases
    # of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, image: np.ndarray) -> PreparedInput:
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image)

    def forward(self, prepared: PreparedInput | Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compatibility alias for :meth:`infer`."""
        return self.infer(prepared)

    def post_process(self, outputs: Mapping[str, np.ndarray], context: ImageContext) -> DetectionResult:
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs, context)

    def __call__(self, image: np.ndarray) -> DetectionResult:
        """Delegate to :meth:`predict`."""
        return self.predict(image)


def _grids(size: int, strides: tuple[int, ...]) -> dict[int, np.ndarray]:
    result = {}
    for stride in strides:
        yv, xv = np.meshgrid(np.arange(size // stride), np.arange(size // stride))
        result[stride] = (((np.stack((yv, xv), axis=2) + 0.5) * stride).reshape(-1, 2)).astype(np.float32)
    return result


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-value))


def _empty_result() -> DetectionResult:
    return DetectionResult(np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32))


def _validate_context(context: ImageContext, binding) -> None:
    """Reject externally supplied contexts that cannot describe this binding."""
    if not isinstance(context, ImageContext):
        raise ValueError("FCOS post_process requires an ImageContext from pre_process.")
    if len(context.original_shape) != 2 or any(type(value) is not int or value <= 0 for value in context.original_shape):
        raise ValueError(f"Invalid original image shape {context.original_shape!r}.")
    if tuple(context.input_shape) != (binding.input_height, binding.input_width):
        raise ValueError(f"Context input shape {context.input_shape!r} does not match the binding.")
    if len(context.resized_shape) != 2 or any(type(value) is not int or value <= 0 for value in context.resized_shape):
        raise ValueError(f"Invalid resized image shape {context.resized_shape!r}.")
    if len(context.pad) != 4 or any(type(value) is not int or value < 0 for value in context.pad):
        raise ValueError(f"Invalid resize padding {context.pad!r}.")
    top, bottom, left, right = context.pad
    resized_h, resized_w = context.resized_shape
    if context.resize_type == 0:
        if context.resized_shape != context.input_shape or context.pad != (0, 0, 0, 0):
            raise ValueError("Direct-resize context must have the full input shape and zero padding.")
    elif context.resize_type == 1:
        if resized_h + top + bottom != binding.input_height or resized_w + left + right != binding.input_width:
            raise ValueError("Letterbox context dimensions and padding do not fill the model input.")
    else:
        raise ValueError(f"Unsupported resize_type in ImageContext: {context.resize_type!r}.")


__all__ = ["DetectionResult", "FCOSTask"]
