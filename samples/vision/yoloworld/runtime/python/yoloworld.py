# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pure YOLOWorld task stages for X5's RGB/text protocol.

This file owns the model end to end: the fixed tensor metadata binding, the
lazy X5 runtime adapter that returns native raw tensors unchanged, and the
``YOLOWorldTask`` stages (preprocess → infer → postprocess → predict).
Published-asset selection and listing live in ``cli.py``.
"""
from __future__ import annotations
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
import cv2
import numpy as np
from utils.py_utils.model_runner import _default_runtime_factory
from utils.py_utils.runtime_meta import MetadataMismatchError, RuntimeMetadata
from samples.vision.yoloworld.runtime.python.cli import ModelSelection


@dataclass(frozen=True)
class ModelBinding:
    selection: ModelSelection
    model_name: str
    image_input_name: str
    text_input_name: str
    score_output_name: str
    box_output_name: str


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | dict[str, Any]) -> ModelBinding:
    """Bind observed metadata to the fixed source protocol without guessing names."""
    # Re-resolve the manifest first: a caller-created ModelSelection (for example
    # a forged same-shape publication row) must not be trusted at this boundary.
    from samples.vision.yoloworld.runtime.python.cli import resolve_selection
    resolved = resolve_selection(
        selection.target, model_path=selection.model_path, asset_id=selection.asset.reference
    )
    if (
        selection.target != resolved.target
        or selection.asset != resolved.asset
        or Path(selection.model_path) != Path(resolved.model_path)
    ):
        raise ValueError("ModelSelection does not match the exact published asset and path.")
    facts = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if facts.model_names and facts.model_name not in facts.model_names:
        raise MetadataMismatchError("Selected runtime model is not among model_names.")
    if len(facts.input_names) != 2 or len(facts.output_names) != 2:
        raise MetadataMismatchError("YOLOWorld requires exactly two inputs and two outputs.")
    for name, shape, dtype in (
        (facts.input_names[0], (1, 3, 640, 640), "float32"),
        (facts.input_names[1], (1, 32, 512, 1), "float32"),
    ):
        if facts.input_shapes.get(name) != shape or facts.input_dtypes.get(name) != dtype:
            raise MetadataMismatchError(f"Input {name!r} is not {dtype}[{','.join(map(str, shape))}].")
    expected = {"score": ((1, 8400, 32), (1, 8400, 32, 1)), "box": ((1, 8400, 4), (1, 8400, 4, 1))}
    score_name = box_name = None
    for name in facts.output_names:
        shape = facts.output_shapes.get(name)
        dtype = facts.output_dtypes.get(name)
        if dtype != "float32":
            raise MetadataMismatchError(f"Output {name!r} must preserve native float32, got {dtype!r}.")
        if shape in expected["score"]:
            if score_name is not None:
                raise MetadataMismatchError("More than one score output matches the source protocol.")
            score_name = name
        elif shape in expected["box"]:
            if box_name is not None:
                raise MetadataMismatchError("More than one box output matches the source protocol.")
            box_name = name
        else:
            raise MetadataMismatchError(f"Output {name!r} has unsupported shape {shape!r}.")
    if score_name is None or box_name is None:
        raise MetadataMismatchError("Missing YOLOWorld score or box output.")
    return ModelBinding(selection, facts.model_name, facts.input_names[0], facts.input_names[1], score_name, box_name)


class RuntimeModelRunner:
    """Load hbm_runtime only on execution and preserve both raw output arrays."""
    def __init__(self, selection: ModelSelection, *, runtime: Any = None,
                 runtime_factory: Callable[[str], Any] | None = None):
        self.selection, self._runtime, self._runtime_factory = selection, runtime, runtime_factory
        self.binding: ModelBinding | None = None
        self.metadata: RuntimeMetadata | None = None

    @property
    def loaded(self) -> bool:
        return self._runtime is not None and self.binding is not None

    def load(self) -> ModelBinding:
        if self.loaded:
            return self.binding  # type: ignore[return-value]
        if self._runtime is None and self._runtime_factory is None:
            # Real execution path: identity and publication gates run before the
            # SDK factory is constructed.  An injected factory is the documented
            # host seam and the only way to skip them.
            from utils.py_utils.platforms import require_execution_target
            from utils.py_utils.assets import verify_asset_file
            require_execution_target(self.selection.target)
            verify_asset_file(self.selection.asset, self.selection.model_path)
        try:
            if self._runtime is None:
                factory = self._runtime_factory or _default_runtime_factory()
                self._runtime = factory(str(self.selection.model_path))
            self.metadata = RuntimeMetadata.from_runtime(self._runtime)
            self.binding = bind_model(self.selection, self.metadata)
            return self.binding
        except Exception:
            self._runtime = None
            self.metadata = None
            self.binding = None
            raise

    def set_scheduling_params(self, *, priority: int | None = None,
                              bpu_cores: list[int] | None = None) -> None:
        binding = self.load()
        if priority is not None and not 0 <= priority <= 255:
            raise ValueError("priority must be in 0..255")
        if bpu_cores is not None and (not bpu_cores or any(core < 0 for core in bpu_cores)):
            raise ValueError("bpu_cores must be a nonempty list of nonnegative indexes")
        if priority is None and bpu_cores is None:
            return
        setter = getattr(self._runtime, "set_scheduling_params", None)
        if not callable(setter):
            raise RuntimeError("Runtime has no set_scheduling_params API")
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {binding.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {binding.model_name: list(bpu_cores)}
        setter(**kwargs)

    def __call__(self, tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        binding = self.load()
        if set(tensors) != {binding.image_input_name, binding.text_input_name}:
            raise MetadataMismatchError("YOLOWorld requires exactly image and text input tensors.")
        image = np.asarray(tensors[binding.image_input_name])
        text = np.asarray(tensors[binding.text_input_name])
        if image.shape != (1, 3, 640, 640) or image.dtype != np.float32 or not np.isfinite(image).all():
            raise MetadataMismatchError("Image input must be finite float32[1,3,640,640].")
        if text.shape != (1, 32, 512, 1) or text.dtype != np.float32 or not np.isfinite(text).all():
            raise MetadataMismatchError("Text input must be finite float32[1,32,512,1].")
        outputs = self._runtime.run({binding.model_name: dict(tensors)})
        if not isinstance(outputs, ABCMapping):
            raise MetadataMismatchError("Runtime returned a non-mapping output.")
        flat = outputs.get(binding.model_name, outputs)
        if not isinstance(flat, ABCMapping) or set(flat) != {binding.score_output_name, binding.box_output_name}:
            raise MetadataMismatchError("Runtime output names changed from the bound protocol.")
        score = np.asarray(flat[binding.score_output_name]); box = np.asarray(flat[binding.box_output_name])
        if score.shape not in ((1, 8400, 32), (1, 8400, 32, 1)) or box.shape not in ((1, 8400, 4), (1, 8400, 4, 1)) or score.dtype != np.float32 or box.dtype != np.float32:
            raise MetadataMismatchError("Runtime outputs must preserve native F32 score/box shapes (logical [1,8400,32] and [1,8400,4], with an optional terminal singleton).")
        if not np.isfinite(score).all() or not np.isfinite(box).all():
            raise MetadataMismatchError("Runtime outputs must be finite.")
        return {binding.score_output_name: score, binding.box_output_name: box}


@dataclass(frozen=True)
class DetectionContext:
    original_height: int
    original_width: int
    scale: float
    prompts: tuple[str, ...]
    class_ids: tuple[int, ...]

@dataclass(frozen=True)
class PreparedInput:
    tensors: Mapping[str, np.ndarray]
    context: DetectionContext

@dataclass(frozen=True)
class DetectionResult:
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    prompts: tuple[str, ...]

class YOLOWorldTask:
    """Open-vocabulary detection from its published selection.

    Construction with a ModelSelection loads the artifact through the
    shared SDK adapter (board identity and artifact bytes verified before
    the SDK is imported) and binds the tensor contract. Host fixtures
    inject an already-constructed runner or a runtime_factory instead.
    No mutable prompt/geometry state is kept between calls.
    """
    def __init__(self, selection, vocabulary: Mapping[str, Sequence[float]],
                 *, score_thres: float = 0.05, nms_thres: float = 0.45,
                 runner=None, runtime_factory=None):
        if not vocabulary:
            raise ValueError("Offline vocabulary must not be empty.")
        # Own a read-only snapshot: a caller that mutates its own embedding array
        # afterwards must not be able to change what later calls send.
        snapshot: dict[str, np.ndarray] = {}
        for key, value in vocabulary.items():
            embedding = np.array(value, dtype=np.float32, copy=True)
            if embedding.shape != (512,) or not np.isfinite(embedding).all():
                raise ValueError("Every offline vocabulary embedding must be finite F32[512].")
            embedding.flags.writeable = False
            snapshot[str(key)] = embedding
        self.class_names = tuple(snapshot)
        self.vocabulary = MappingProxyType(snapshot)
        self.score_thres, self.nms_thres = float(score_thres), float(nms_thres)
        if not 0 <= self.score_thres <= 1 or not 0 <= self.nms_thres <= 1:
            raise ValueError("score_thres and nms_thres must be in [0,1].")
        if runner is None:
            if not isinstance(selection, ModelSelection):
                raise TypeError("Pass a ModelSelection from resolve_selection, or inject runner=/runtime_factory=.")
            if runtime_factory is None:
                runner = RuntimeModelRunner(selection)
            else:
                runner = RuntimeModelRunner(selection, runtime_factory=runtime_factory)
        self.runner = runner
        self.binding: ModelBinding = runner.load()

    def set_scheduling_params(self, *, priority: int | None = None,
                              bpu_cores: list[int] | None = None) -> None:
        """Apply explicit scheduling values to the loaded board runtime."""
        self.runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

    def _prompts(self, prompts: Sequence[str]) -> tuple[tuple[str, ...], tuple[int, ...], np.ndarray]:
        if isinstance(prompts, str):
            raise TypeError("prompts must be a sequence of strings, not one comma string")
        values = tuple(str(p).strip() for p in prompts)
        if not values or any(not p for p in values):
            raise ValueError("At least one nonempty prompt is required; empty prompts are rejected.")
        if len(values) > 32:
            raise ValueError("YOLOWorld accepts at most 32 prompts per call.")
        ids = tuple(self.class_names.index(p) if p in self.vocabulary else -1 for p in values)
        if -1 in ids:
            missing = values[ids.index(-1)]
            raise KeyError(f"Prompt {missing!r} is absent from offline vocabulary.")
        rows = [self.vocabulary[p] for p in values]
        while len(rows) < 32:
            rows.append(rows[-1])
            ids = ids + (ids[-1],)
        return values, ids, np.asarray(rows, dtype=np.float32).reshape(1, 32, 512, 1)

    def preprocess(self, image: np.ndarray, prompts: Sequence[str]) -> PreparedInput:
        """Letterbox one BGR image to the 640 canvas, build the RGB tensor and tokenized prompts, and return prepared tensors with the per-call context."""

        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("image must be a nonempty HxWx3 BGR array")
        if image.dtype not in (np.uint8, np.float32):
            raise ValueError("image must be uint8 or float32")
        image_f = image.astype(np.float32, copy=False)
        if not np.isfinite(image_f).all() or image_f.min() < 0 or image_f.max() > 255:
            raise ValueError("image must contain finite values in [0,255]")
        names, ids, text = self._prompts(prompts)
        h, w = image.shape[:2]
        scale = max(h, w) / 640.0
        resized = cv2.resize(image, (0, 0), fx=1.0 / scale, fy=1.0 / scale)
        canvas = np.zeros((640, 640, 3), dtype=np.float32)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        rgb = canvas[:, :, ::-1].transpose(2, 0, 1)[None].copy()
        context = DetectionContext(h, w, scale, names, ids)
        return PreparedInput(MappingProxyType({self.binding.image_input_name: rgb,
                                                 self.binding.text_input_name: text}), context)

    def infer(self, prepared: PreparedInput | Mapping[str, np.ndarray]):
        """Execute the bound runner on prepared tensors and return the raw output arrays."""

        tensors = prepared.tensors if isinstance(prepared, PreparedInput) else prepared
        return self.runner(tensors)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, threshold: float) -> list[int]:
        keep: list[int] = []
        for cls in np.unique(classes):
            indices = np.flatnonzero(classes == cls)
            order = indices[np.argsort(scores[indices])[::-1]]
            while len(order):
                current = int(order[0]); keep.append(current)
                if len(order) == 1: break
                rest = order[1:]
                xx1 = np.maximum(boxes[current, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
                xx2 = np.minimum(boxes[current, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
                inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
                area = np.maximum(0, boxes[current, 2]-boxes[current, 0]) * np.maximum(0, boxes[current, 3]-boxes[current, 1])
                rest_area = np.maximum(0, boxes[rest, 2]-boxes[rest, 0]) * np.maximum(0, boxes[rest, 3]-boxes[rest, 1])
                iou = inter / (area + rest_area - inter + 1e-9)
                order = rest[iou < threshold]
        return keep

    def postprocess(self, outputs: Mapping[str, np.ndarray], context: DetectionContext) -> DetectionResult:
        """Decode raw outputs into boxes, scores and prompt names using the per-call context."""

        score_raw = np.asarray(outputs[self.binding.score_output_name]); box_raw = np.asarray(outputs[self.binding.box_output_name])
        if score_raw.shape not in ((1, 8400, 32), (1, 8400, 32, 1)) or box_raw.shape not in ((1, 8400, 4), (1, 8400, 4, 1)) or score_raw.dtype != np.float32 or box_raw.dtype != np.float32:
            raise ValueError("Raw outputs must preserve native F32 logical score/box shapes.")
        if score_raw.ndim == 4: score_raw = score_raw[..., 0]
        if box_raw.ndim == 4: box_raw = box_raw[..., 0]
        slots = np.argmax(score_raw[0], axis=1)
        scores = score_raw[0, np.arange(8400), slots]
        mask = scores >= self.score_thres
        if not np.any(mask):
            return DetectionResult(np.empty((0,4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32), context.prompts)
        boxes = box_raw[0, mask].copy(); scores = scores[mask].copy(); slots = slots[mask].astype(np.int32)
        keep = self._nms(boxes, scores, slots, self.nms_thres)
        boxes, scores, slots = boxes[keep], scores[keep], slots[keep]
        boxes *= context.scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, context.original_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, context.original_height)
        ids = np.asarray([context.class_ids[int(slot)] for slot in slots], dtype=np.int32)
        return DetectionResult(boxes, scores, ids, context.prompts)

    def predict(self, image: np.ndarray, prompts: Sequence[str]) -> DetectionResult:
        """Chain preprocess, infer and postprocess for one image and its prompts."""

        prepared = self.preprocess(image, prompts)
        return self.postprocess(self.infer(prepared), prepared.context)

    def pre_process(self, image: np.ndarray, prompts: Sequence[str]) -> PreparedInput:
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image, prompts)

    def forward(self, prepared: PreparedInput | Mapping[str, np.ndarray]):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(prepared)

    def post_process(self, outputs: Mapping[str, np.ndarray], context: DetectionContext) -> DetectionResult:
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs, context)
