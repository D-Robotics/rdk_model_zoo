# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""R3D-18 video classification: load, preprocess, infer, postprocess, predict.

``R3D18Classifier`` owns the fixed five-dimensional S100 contract end to end:
construction loads the model through the sample's lazy runner, and each
``predict`` call runs preprocess -> infer -> postprocess visible in this
file. Catalog selection and reporting live in ``cli.py``; Kinetics label
decoding lives in ``labels.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from utils.py_utils.classification import ClassificationResult, topk_from_scores
from utils.py_utils.cls_binding import score_vector_shape
from utils.py_utils.model_runner import _default_runtime_factory
from utils.py_utils.runtime_meta import MetadataMismatchError, RuntimeMetadata, canonicalise_dtype

INPUT_SHAPE = (1, 3, 16, 112, 112)
CLASS_COUNT = 400


@dataclass(frozen=True)
class ModelBinding:
    """Validated R3D-18 tensor protocol for the S100 artifact.

    Attributes:
        selection: The manifest-backed selection this binding was built from.
        model_name: Single submodel name used for SDK run calls.
        input_name: Bound five-dimensional F32 input tensor name.
        input_shape: Fixed input shape ``(1, 3, 16, 112, 112)``.
        output_name: Bound F32 score tensor name.
        output_shape: Exact observed score tensor shape.
        input_dtype: Fixed input dtype ``float32``.
        output_dtype: Fixed output dtype ``float32``.
        output_transform: Fixed ``raw_f32``; scores are not dequantized.
    """

    selection: "Any"
    model_name: str
    input_name: str
    input_shape: tuple[int, ...]
    output_name: str
    output_shape: tuple[int, ...]
    input_dtype: str
    output_dtype: str
    output_transform: str = "raw_f32"


def bind_model(selection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    """Validate the fixed source R3D-18 tensor protocol.

    Args:
        selection: Manifest-backed selection whose asset and path must match.
        metadata: Board-observed metadata mapping or ``RuntimeMetadata``.

    Returns:
        ModelBinding: Validated tensor names and shapes for the 5-D contract.

    Raises:
        ValueError: The selection is not the exact published S100 asset.
        MetadataMismatchError: Tensor names, shapes, or dtypes violate the
            contract.
    """
    import importlib

    # ``3dresnet`` starts with a digit: package imports go through importlib.
    _cli = importlib.import_module("samples.vision.3dresnet.runtime.python.cli")

    if selection.asset.reference != _cli.ASSET_ID or selection.target != "s100":
        raise ValueError("3DResNet binding received an unpublished selection.")
    published = _cli.resolve_selection(
        selection.target,
        asset_id=selection.asset.reference,
        model_path=selection.model_path,
    )
    if published.asset != selection.asset:
        raise ValueError("3DResNet selection publication facts do not match the manifest.")
    facts = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if len(facts.input_names) != 1 or len(facts.output_names) != 1:
        raise MetadataMismatchError("R3D-18 requires exactly one input and one output tensor.")
    input_name = facts.input_names[0]
    output_name = facts.output_names[0]
    if facts.input_shapes.get(input_name) != INPUT_SHAPE:
        raise MetadataMismatchError(
            f"R3D-18 input {input_name!r} must have shape {INPUT_SHAPE}; "
            f"got {facts.input_shapes.get(input_name)}."
        )
    if canonicalise_dtype(facts.input_dtypes.get(input_name)) != "float32":
        raise MetadataMismatchError("R3D-18 input must be float32.")
    output_shape = facts.output_shapes.get(output_name)
    if output_shape is None or not score_vector_shape(output_shape, CLASS_COUNT):
        raise MetadataMismatchError(
            f"R3D-18 output {output_name!r} must squeeze to 400 scores; got {output_shape}."
        )
    if canonicalise_dtype(facts.output_dtypes.get(output_name)) != "float32":
        raise MetadataMismatchError("R3D-18 output must be float32 scores.")
    return ModelBinding(
        selection=selection,
        model_name=facts.model_name,
        input_name=input_name,
        input_shape=INPUT_SHAPE,
        output_name=output_name,
        output_shape=tuple(output_shape),
        input_dtype="float32",
        output_dtype="float32",
    )


class RuntimeModelRunner:
    """Load and call one metadata-bound R3D-18 model lazily.

    The board identity gate runs before the SDK import on the real path; an
    injected ``runtime``/``runtime_factory`` is the documented host seam.
    """

    def __init__(
        self,
        selection,
        *,
        runtime: Any = None,
        runtime_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.selection = selection
        self._runtime = runtime
        self._runtime_factory = runtime_factory
        self.metadata: RuntimeMetadata | None = None
        self.binding: ModelBinding | None = None

    @property
    def loaded(self) -> bool:
        return self._runtime is not None and self.binding is not None

    @property
    def runtime(self) -> Any:
        if self._runtime is None:
            raise RuntimeError("Runtime has not been loaded; call load() first.")
        return self._runtime

    def load(self) -> ModelBinding:
        if self.loaded:
            return self.binding  # type: ignore[return-value]
        if self._runtime is None and self._runtime_factory is None:
            from utils.py_utils.platforms import require_execution_target

            require_execution_target(self.selection.target)
        try:
            if self._runtime is None:
                factory = self._runtime_factory or _default_runtime_factory()
                self._runtime = factory(str(self.selection.model_path))
            self.metadata = RuntimeMetadata.from_runtime(self._runtime)
            self.binding = bind_model(self.selection, self.metadata)
        except Exception:
            self._runtime = None
            self.metadata = None
            self.binding = None
            raise
        return self.binding

    def set_scheduling_params(self, *, priority: int | None = None, bpu_cores: list[int] | None = None) -> None:
        binding = self.load()
        if priority is not None and not 0 <= priority <= 255:
            raise ValueError("priority must be between 0 and 255.")
        if bpu_cores is not None and (not bpu_cores or any(core < 0 for core in bpu_cores)):
            raise ValueError("bpu_cores must be a nonempty list of nonnegative indexes.")
        if priority is None and bpu_cores is None:
            return
        setter = getattr(self.runtime, "set_scheduling_params", None)
        if not callable(setter):
            raise RuntimeError("The installed runtime does not expose scheduling parameters.")
        kwargs: dict[str, Any] = {}
        if priority is not None:
            kwargs["priority"] = {binding.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {binding.model_name: list(bpu_cores)}
        setter(**kwargs)

    def __call__(self, tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        binding = self.load()
        if not isinstance(tensors, Mapping) or set(tensors) != {binding.input_name}:
            raise MetadataMismatchError(f"R3D-18 requires exactly one input named {binding.input_name!r}.")
        value = np.asarray(tensors[binding.input_name])
        if (
            value.shape != binding.input_shape
            or value.dtype != np.dtype(binding.input_dtype)
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
        ):
            raise MetadataMismatchError("R3D-18 input must be finite contiguous float32 (1,3,16,112,112).")
        outputs = self.runtime.run({binding.model_name: dict(tensors)})
        if not isinstance(outputs, Mapping):
            raise MetadataMismatchError("Runtime returned a non-mapping output.")
        flat = outputs.get(binding.model_name, outputs)
        if not isinstance(flat, Mapping) or set(flat) != {binding.output_name}:
            raise MetadataMismatchError("Runtime output must contain exactly the bound score tensor.")
        output = np.asarray(flat[binding.output_name])
        if output.shape != binding.output_shape or output.dtype != np.dtype(binding.output_dtype):
            raise MetadataMismatchError(
                f"Runtime output {binding.output_name!r} differs from bound shape/dtype."
            )
        if not np.isfinite(output).all():
            raise MetadataMismatchError("R3D-18 output contains NaN or infinity.")
        return {binding.output_name: output}


@dataclass(frozen=True)
class VideoContext:
    """Source clip identity for one prepared call.

    Attributes:
        original_shape: Source clip shape before preparation.
        original_dtype: Source clip dtype name.
        input_shape: Bound tensor shape the clip was cast to.
    """

    original_shape: tuple[int, ...]
    original_dtype: str
    input_shape: tuple[int, ...]


@dataclass(frozen=True)
class PreparedInput:
    """One R3D-18 tensor mapping and its independent clip context.

    Attributes:
        tensors: Input-name to contiguous float32 5-D tensor mapping.
        context: Frozen source identity of this call only.
    """

    tensors: Mapping[str, np.ndarray]
    context: VideoContext


def prepare_clip(clip: np.ndarray, binding: ModelBinding) -> PreparedInput:
    """Validate one normalized clip and cast it to the bound float32 tensor.

    Args:
        clip: Numeric array shaped exactly ``(1, 3, 16, 112, 112)``; already
            normalized and not decoded or resized here.
        binding: Validated R3D-18 binding supplying tensor names and shapes.

    Returns:
        PreparedInput: Input-name mapping of one contiguous float32 tensor,
        plus this call's frozen clip context.

    Raises:
        ValueError: The clip is not a numeric array of the exact shape, or
            contains nonfinite values after the float32 cast.
    """
    if not isinstance(clip, np.ndarray):
        raise ValueError("Expected a NumPy video clip.")
    if clip.shape != INPUT_SHAPE:
        raise ValueError(f"Expected input shape {INPUT_SHAPE}, got {clip.shape}.")
    if not np.issubdtype(clip.dtype, np.number):
        raise ValueError(f"Video clip must be numeric, got {clip.dtype}.")
    tensor = np.ascontiguousarray(clip.astype(np.float32, copy=True))
    if not np.isfinite(tensor).all():
        raise ValueError("Video clip contains NaN or infinity.")
    if tensor.shape != binding.input_shape or tensor.dtype != np.dtype(binding.input_dtype):
        raise ValueError("Prepared video clip does not match the bound input.")
    return PreparedInput(
        tensors={binding.input_name: tensor},
        context=VideoContext(tuple(clip.shape), str(clip.dtype), tuple(tensor.shape)),
    )


class R3D18Classifier:
    """Classify one normalized video clip with a compiled R3D-18 model.

    The constructor accepts a resolved selection; see __init__ for the
    injection seam. Construction loads the model immediately.

    Attributes:
        runner (RuntimeModelRunner): Lazy sample runner used by infer.
        binding (ModelBinding): Validated tensor names, shapes, and dtypes.
        top_k (int): Number of ranked classes returned per clip.
        labels: Optional class names keyed by index.
    """

    def __init__(
        self,
        selection,
        *,
        top_k: int = 5,
        labels: Optional[Mapping[int, str] | Sequence[str]] = None,
        runner: Optional[RuntimeModelRunner] = None,
    ) -> None:
        """Load the compiled model and validate its classification settings.

        Args:
            selection: Manifest-backed selection from ``cli.resolve_selection``.
            top_k: Number of results in [1, 400]; defaults to 5.
            labels: Optional class-name mapping or sequence.
            runner: Optional injected runner (host-test seam); defaults to
                the lazy sample runner with the board-identity gate.

        Returns:
            None.

        Raises:
            TypeError: The injected runner is not callable.
            ValueError: top_k is out of range, or the selection/metadata
                violates the contract.
            RuntimeError: Board identity or SDK loading fails.
        """
        self.runner = runner if runner is not None else RuntimeModelRunner(selection)
        if not callable(self.runner):
            raise TypeError("runner must be callable.")
        self.binding = self.runner.load()
        if not isinstance(top_k, (int, np.integer)) or isinstance(top_k, bool) or not 1 <= int(top_k) <= CLASS_COUNT:
            raise ValueError(f"top_k must be between 1 and {CLASS_COUNT}.")
        self.top_k = int(top_k)
        self.labels = labels

    def preprocess(self, clip: np.ndarray) -> PreparedInput:
        """Validate one clip and cast it to the bound float32 input tensor.

        Args:
            clip: Numeric array shaped ``(1, 3, 16, 112, 112)``; already
                normalized; not decoded or resized here.

        Returns:
            PreparedInput: Input-name mapping of one contiguous float32
            tensor, plus this call's frozen clip context.

        Raises:
            ValueError: The clip shape, dtype, or values are invalid.
        """
        return prepare_clip(clip, self.binding)

    def infer(self, prepared: "PreparedInput | Mapping[str, np.ndarray]") -> Mapping[str, np.ndarray]:
        """Execute one inference call with the prepared tensor.

        Args:
            prepared: ``PreparedInput`` from preprocess, or its tensors mapping.

        Returns:
            Mapping[str, np.ndarray]: Raw owned float32 score tensor keyed by
            the bound output name. No softmax, Top-K, or I/O happens here.

        Raises:
            MetadataMismatchError: Input names, shape, dtype, contiguity, or
                the runtime output structure violate the binding.
            RuntimeError: SDK execution fails.
        """
        tensors = prepared.tensors if isinstance(prepared, PreparedInput) else prepared
        return self.runner(tensors)

    def postprocess(self, outputs: Mapping[str, np.ndarray], *, top_k: int | None = None) -> ClassificationResult:
        """Apply source-compatible softmax and select the highest-ranked classes.

        Args:
            outputs: Raw output mapping returned by infer, containing exactly
                the bound finite float32 score tensor.
            top_k: Optional per-call override of the constructed top_k.

        Returns:
            ClassificationResult: Ranked int64 class_ids and float32 scores
            with a matching label tuple; scores descend and ties prefer
            lower class IDs.

        Raises:
            ValueError: Output container, shape, dtype, finiteness, or the
                top_k override is invalid.
        """
        if not isinstance(outputs, Mapping) or set(outputs) != {self.binding.output_name}:
            raise ValueError("R3D-18 raw outputs must contain exactly the bound score tensor.")
        raw = np.asarray(outputs[self.binding.output_name])
        if raw.shape != self.binding.output_shape:
            raise ValueError(
                f"R3D-18 output shape must be {self.binding.output_shape}, got {raw.shape}."
            )
        if raw.dtype != np.dtype(self.binding.output_dtype):
            raise ValueError(f"R3D-18 output dtype must be {self.binding.output_dtype}, got {raw.dtype}.")
        if not np.isfinite(raw).all():
            raise ValueError("R3D-18 output contains NaN or infinity.")
        selected_k = self.top_k if top_k is None else top_k
        if not isinstance(selected_k, (int, np.integer)) or isinstance(selected_k, bool) or not 1 <= int(selected_k) <= CLASS_COUNT:
            raise ValueError(f"top_k must be between 1 and {CLASS_COUNT}.")
        return topk_from_scores(raw, int(selected_k), self.labels, softmax=True)

    def predict(self, clip: np.ndarray, *, top_k: int | None = None) -> ClassificationResult:
        """Run preprocessing, inference, and postprocessing for one clip.

        Args:
            clip: Numeric array shaped ``(1, 3, 16, 112, 112)``.
            top_k: Optional per-call override of the constructed top_k.

        Returns:
            ClassificationResult: Ranked classes; see postprocess.

        Raises:
            ValueError: Clip data, tensors, or scores are invalid.
            MetadataMismatchError: Tensor structure violates the binding.
            RuntimeError: SDK execution fails.
        """
        prepared = self.preprocess(clip)
        return self.postprocess(self.infer(prepared.tensors), top_k=top_k)

    def set_scheduling_params(self, *, priority=None, bpu_cores=None) -> None:
        """Apply scheduling options to the loaded board runtime.

        Args:
            priority: Optional integer in [0, 255]; None leaves it unchanged.
            bpu_cores: Optional non-empty list of nonnegative BPU core indexes.

        Returns:
            None.

        Raises:
            ValueError: Priority or a core index is out of range.
            RuntimeError: The SDK cannot apply the scheduling options.
        """
        self.runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

    def pre_process(self, clip: np.ndarray) -> PreparedInput:
        """Delegate to preprocess with the same input and error contract."""
        return self.preprocess(clip)

    def forward(self, prepared: "PreparedInput | Mapping[str, np.ndarray]") -> Mapping[str, np.ndarray]:
        """Delegate to infer with the same input and error contract."""
        return self.infer(prepared)

    def post_process(self, outputs: Mapping[str, np.ndarray], *, top_k: int | None = None) -> ClassificationResult:
        """Delegate to postprocess with the same input and error contract."""
        return self.postprocess(outputs, top_k=top_k)

    def __call__(self, clip: np.ndarray, *, top_k: int | None = None) -> ClassificationResult:
        """Delegate to predict with the same input and error contract."""
        return self.predict(clip, top_k=top_k)


__all__ = ["ClassificationResult", "PreparedInput", "R3D18Classifier",
           "RuntimeModelRunner", "VideoContext", "bind_model", "prepare_clip"]
