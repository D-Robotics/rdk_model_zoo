# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""FCOS readable task: tensor contract, runner, preprocess, infer, postprocess, predict.

``fcos.py`` owns the model end to end: the observed-metadata tensor binding
(all fifteen FCOS outputs resolved by exact shape family), the lazy X5
runtime adapter, and the ``FCOSTask`` stages (``preprocess`` → ``infer`` →
``postprocess`` → ``predict``). Published asset identity/selection and
rendering live in ``cli.py``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from utils.py_utils.image import bgr_to_nv12_planes
from utils.py_utils.quantization import apply_output_transform
from utils.py_utils.runtime_meta import RuntimeMetadata, canonicalise_dtype
from samples.vision.fcos.runtime.python.cli import (
    BindingError,
    FCOSContract,
    ModelSelection,
    _records,
)

# ======================================================================
# Observed-metadata tensor contract: fifteen outputs bound by exact shape
# family, quantization descriptors retained for post-processing.
# ======================================================================

_ALLOWED_OUTPUT_DTYPES = {"float16", "float32", "int8", "uint8", "int16", "int32"}


@dataclass(frozen=True)
class ModelBinding:
    """Observed runtime metadata bound to one exact FCOS artifact."""

    selection: ModelSelection
    metadata: RuntimeMetadata
    model_name: str
    input_names: tuple[str, ...]
    input_shapes: Mapping[str, tuple[int, ...]]
    output_names: tuple[str, ...]
    output_shapes: Mapping[str, tuple[int, ...]]
    output_dtypes: Mapping[str, str]
    output_quants: Mapping[str, Any]
    cls_output_names: tuple[str, ...]
    box_output_names: tuple[str, ...]
    center_output_names: tuple[str, ...]

    @property
    def contract(self) -> FCOSContract:
        """Return the static contract used for binding."""
        return self.selection.contract

    @property
    def input_width(self) -> int:
        return self.contract.input_width

    @property
    def input_height(self) -> int:
        return self.contract.input_height

    def validate_inputs(self, tensors: Mapping[str, np.ndarray]) -> None:
        """Validate packed NV12 tensors without changing their identity."""
        import numpy as np

        if tuple(tensors) != self.input_names:
            raise BindingError(f"Input names {tuple(tensors)!r} != {self.input_names!r}.")
        value = tensors[self.input_names[0]]
        if not isinstance(value, np.ndarray):
            raise BindingError("Packed NV12 input must be a NumPy array.")
        expected = self.input_height * self.input_width * 3 // 2
        if value.ndim != 1 or value.size != expected or value.dtype != np.uint8:
            raise BindingError(
                f"Packed NV12 input must be uint8 shape ({expected},), got "
                f"{value.shape} {value.dtype}."
            )
        if not value.flags.c_contiguous:
            raise BindingError("Packed NV12 input must be contiguous.")

    def validate_outputs(self, outputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Validate raw output containers and return the same ndarray objects.

        Matching is by exact name set, never by insertion order: the board
        ``hbm_runtime`` ``run()`` mapping does not preserve
        ``metadata.output_names`` order while the fifteen names themselves
        are identical (X5 evidence 2026-09-24).  Missing and extra names are
        both rejected, and every bound name is checked against the binding's
        shape, dtype, and finiteness.  The mapping and each ndarray stay
        caller-owned and identity-stable; roles are resolved only from the
        binding's own name tuples, never by iterating the caller's dict.
        """
        import numpy as np

        if not isinstance(outputs, Mapping):
            raise BindingError("Runtime output must be a name→ndarray mapping.")
        observed = set(outputs)
        bound = set(self.output_names)
        missing = sorted(bound - observed)
        extra = sorted(observed - bound)
        if missing or extra:
            raise BindingError(
                f"Output names must match the binding exactly; "
                f"missing={missing}, unexpected={extra}."
            )
        for name in self.output_names:
            value = outputs[name]
            if not isinstance(value, np.ndarray):
                raise BindingError(f"Output {name!r} must be a NumPy array.")
            if tuple(value.shape) != self.output_shapes[name]:
                raise BindingError(
                    f"Output {name!r} shape {value.shape} != {self.output_shapes[name]}."
                )
            if canonicalise_dtype(value.dtype) != self.output_dtypes[name]:
                raise BindingError(
                    f"Output {name!r} dtype {value.dtype} != {self.output_dtypes[name]}."
                )
            if not np.all(np.isfinite(value.astype(np.float32, copy=False))):
                raise BindingError(f"Output {name!r} contains non-finite values.")
        # The raw output mapping and each ndarray remain caller-owned and
        # identity-stable; validation is deliberately observational.
        return outputs  # type: ignore[return-value]


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    """Bind all fifteen observed tensors to the selected FCOS contract.

    Output names are not guessed.  Their roles are resolved only by the exact
    source shape family for each stride and channel count; every output keeps
    its runtime quantization descriptor for post-processing.
    """
    records = {record.asset_id: record for record in _records()}
    if selection.asset_id not in records or records[selection.asset_id].variant != selection.variant:
        raise BindingError("Selection does not match a current FCOS manifest asset.")
    facts = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if tuple(facts.model_names) != (facts.model_name,):
        raise BindingError("FCOS artifact must expose exactly one selected runtime model.")
    if len(facts.input_names) != 1:
        raise BindingError("FCOS X5 expects one packed input tensor.")
    input_name = facts.input_names[0]
    if tuple(facts.input_shapes.get(input_name, ())) != (1, 3, selection.contract.input_height, selection.contract.input_width):
        raise BindingError("Runtime input shape does not match the selected FCOS variant.")
    if canonicalise_dtype(facts.input_dtypes.get(input_name)) != "nv12":
        raise BindingError("FCOS input metadata must report NV12.")
    if len(facts.output_names) != 15 or set(facts.output_names) != set(facts.output_shapes):
        raise BindingError("FCOS metadata must expose exactly fifteen named outputs.")
    cls: list[str] = []
    box: list[str] = []
    center: list[str] = []
    used: set[str] = set()
    size = selection.contract.input_height
    for names, channels in ((cls, 80), (box, 4), (center, 1)):
        for stride in selection.contract.strides:
            expected = (1, size // stride, size // stride, channels)
            matches = [name for name in facts.output_names if name not in used and tuple(facts.output_shapes.get(name, ())) == expected]
            if len(matches) != 1:
                raise BindingError(f"Expected exactly one FCOS output with shape {expected}, found {matches}.")
            name = matches[0]
            used.add(name)
            names.append(name)
    if used != set(facts.output_names):
        raise BindingError("FCOS metadata contains an unclassified output tensor.")
    for name in facts.output_names:
        dtype = canonicalise_dtype(facts.output_dtypes.get(name))
        if dtype not in _ALLOWED_OUTPUT_DTYPES:
            raise BindingError(f"Unsupported FCOS output dtype for {name!r}: {dtype!r}.")
        if name not in facts.output_quants or not _is_quant_descriptor(facts.output_quants[name]):
            raise BindingError(f"FCOS output {name!r} is missing its quantization descriptor.")
    return ModelBinding(
        selection=selection,
        metadata=facts,
        model_name=facts.model_name,
        input_names=facts.input_names,
        input_shapes=facts.input_shapes,
        output_names=facts.output_names,
        output_shapes=facts.output_shapes,
        output_dtypes={name: canonicalise_dtype(facts.output_dtypes[name]) or "" for name in facts.output_names},
        output_quants=facts.output_quants,
        cls_output_names=tuple(cls),
        box_output_names=tuple(box),
        center_output_names=tuple(center),
    )


def _is_quant_descriptor(value: Any) -> bool:
    """Accept only runtime-like descriptors that source dequant can inspect."""
    return value is not None and all(hasattr(value, key) for key in ("quant_type", "scale", "zero_point"))


# ======================================================================
# Lazy X5 runtime adapter.
# ======================================================================

class RuntimeUnavailableError(RuntimeError):
    """The board-only ``hbm_runtime`` package is unavailable."""


class RuntimeModelRunner:
    """Load the board SDK only after an execution call requests it."""

    def __init__(self, selection: ModelSelection, *, runtime_factory: Callable[[str], Any] | None = None, runtime: Any = None):
        self.selection = selection
        self._factory = runtime_factory
        self._runtime = runtime
        self.binding: ModelBinding | None = None
        self.metadata: RuntimeMetadata | None = None

    @property
    def loaded(self) -> bool:
        """Whether runtime and binding metadata have both been loaded."""
        return self._runtime is not None and self.binding is not None

    def load(self) -> ModelBinding:
        """Load, observe, and bind one exact selected model."""
        if self.loaded:
            return self.binding  # type: ignore[return-value]
        if self._runtime is None:
            if self._factory is None:
                from utils.py_utils.platforms import require_execution_target
                from utils.py_utils.assets import resolve_asset, verify_asset_file

                require_execution_target(self.selection.target)
                verify_asset_file(resolve_asset(self.selection.asset_id), self.selection.model_path)
            factory = self._factory or _default_runtime_factory()
            self._runtime = factory(str(self.selection.model_path))
        try:
            self.metadata = RuntimeMetadata.from_runtime(self._runtime)
            self.binding = bind_model(self.selection, self.metadata)
        except Exception:
            self._runtime = None
            self.metadata = None
            self.binding = None
            raise
        return self.binding

    def set_scheduling_params(self, *, priority: int | None = None, bpu_cores: list[int] | None = None) -> None:
        """Apply source scheduling parameters after metadata binding."""
        binding = self.load()
        if priority is not None and not 0 <= priority <= 255:
            raise ValueError("priority must be between 0 and 255.")
        if bpu_cores is not None and any((not isinstance(core, int) or core < 0) for core in bpu_cores):
            raise ValueError("bpu_cores must contain non-negative integers.")
        if priority is None and bpu_cores is None:
            return
        setter = getattr(self._runtime, "set_scheduling_params", None)
        if not callable(setter):
            raise RuntimeError("hbm_runtime does not expose set_scheduling_params.")
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {binding.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {binding.model_name: bpu_cores}
        setter(**kwargs)

    def __call__(self, tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run one validated packed input and return raw output arrays."""
        binding = self.load()
        binding.validate_inputs(tensors)
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("Runtime has not been loaded.")
        outputs = runtime.run({binding.model_name: dict(tensors)})
        if not isinstance(outputs, Mapping):
            raise BindingError("Runtime returned a non-mapping output.")
        flat = outputs.get(binding.model_name, outputs)
        if not isinstance(flat, Mapping):
            raise BindingError("Runtime returned a non-mapping model output.")
        return binding.validate_outputs(flat)


def _default_runtime_factory() -> Callable[[str], Any]:
    try:
        module = importlib.import_module("hbm_runtime")
    except ImportError as exc:
        raise RuntimeUnavailableError("hbm_runtime is board-only; host help/list/dry-run do not need it.") from exc
    factory = getattr(module, "HB_HBMRuntime", None)
    if not callable(factory):
        raise RuntimeUnavailableError("hbm_runtime does not expose HB_HBMRuntime.")
    return factory


# ======================================================================
# The readable FCOS task stages.
# ======================================================================


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


__all__ = ["DetectionResult", "FCOSTask", "ModelBinding", "RuntimeModelRunner", "RuntimeUnavailableError", "bind_model"]
