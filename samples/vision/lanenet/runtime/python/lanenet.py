# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""LaneNet lane segmentation: load, preprocess, infer, postprocess, predict.

``LaneNetSegmenter`` owns the fixed 256x512 RGB contract end to end:
construction loads the model through the shared lazy transport, and each
``predict`` call runs preprocess -> infer -> postprocess visible in this
file. The three stages return raw embeddings and model-grid binary labels,
not lane IDs. Catalog selection and presentation live in ``cli.py``; the
source input arithmetic shared with conversion calibration lives here as
``image_to_tensor``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.py_utils.platforms import require_execution_target
from utils.py_utils.runtime_meta import MetadataMismatchError, RuntimeMetadata
from utils.py_utils.single_array_runner import NamedArrayRunner

from samples.vision.lanenet.runtime.python.cli import ModelSelection


def image_to_tensor(image):
    """BGR uint8 HWC → the source RGB/ImageNet float32 NCHW input tensor.

    INTER_AREA stretch to (512, 256) preserves the source input policy; the
    explicit-calibration conversion script imports this same function.
    """
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or min(image.shape[:2]) <= 0
    ):
        raise ValueError("Expected nonempty BGR uint8 HWC image")
    rgb = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        (512, 256),
        interpolation=cv2.INTER_AREA,
    )
    chw = (rgb.astype(np.float32) / 255).transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    return np.ascontiguousarray(((chw - mean) / std)[None])


@dataclass(frozen=True)
class ModelBinding:
    """Validated fixed input and named embedding/binary output roles.

    Attributes:
        selection: The manifest-backed selection this binding was built from.
        metadata: Board-observed model metadata validated against the contract.
        input_name: Bound normalized RGB F32 input tensor name.
        embedding_name: Bound instance-embedding output tensor name.
        binary_name: Bound binary lane-label output tensor name.
    """

    selection: ModelSelection
    metadata: RuntimeMetadata
    input_name: str
    embedding_name: str
    binary_name: str

    @property
    def model_name(self) -> str:
        """Return the single submodel name declared by the artifact."""
        return self.metadata.model_name


def bind_model(
    selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]
) -> ModelBinding:
    """Bind required roles by exact names; retain declared auxiliary tensors.

    Args:
        selection: Manifest-backed selection whose asset and path must match.
        metadata: Board-observed metadata mapping or ``RuntimeMetadata``.

    Returns:
        ModelBinding: Validated input and embedding/binary output roles.

    Raises:
        BindingError: The selection differs from the exact manifest contract.
        MetadataMismatchError: Tensor names, shapes, or dtypes violate the
            contract.

    Notes:
        Source prose mentions a third output without identifying it. No name
        or semantic is invented for such outputs; the runner preserves
        observed raw IO.
    """
    from samples.vision.lanenet.runtime.python.cli import BindingError, resolve_selection

    expected = resolve_selection(
        selection.target,
        asset_id=selection.asset.reference,
        model_path=selection.model_path if selection.explicit_model_path else None,
    )
    if expected != selection:
        raise BindingError("Selection differs from exact manifest contract")
    meta = (
        metadata
        if isinstance(metadata, RuntimeMetadata)
        else RuntimeMetadata.from_mapping(metadata)
    )
    if len(meta.model_names) != 1 or len(meta.input_names) != 1:
        raise MetadataMismatchError("Expected exactly one model and one input")
    inp = meta.input_names[0]
    if (
        meta.input_shapes.get(inp) != (1, 3, 256, 512)
        or meta.input_dtypes.get(inp) != "float32"
    ):
        raise MetadataMismatchError("Expected float32 NCHW RGB [1,3,256,512]")
    names = meta.output_names
    required = {"instance_seg_logits", "binary_seg_pred"}
    if len(set(names)) != len(names) or not required.issubset(names):
        raise MetadataMismatchError(
            "Required named embedding and binary outputs are missing or duplicated"
        )
    for name in names:
        shape = meta.output_shapes.get(name, ())
        if not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise MetadataMismatchError(f"Invalid fixed output shape: {name}")
        if meta.output_dtypes.get(name) not in (
            "float16",
            "float32",
            "int8",
            "uint8",
            "int16",
            "int32",
            "int64",
        ):
            raise MetadataMismatchError(f"Unsupported raw output dtype: {name}")
    embedding, binary = "instance_seg_logits", "binary_seg_pred"
    if (
        meta.output_shapes[embedding] != (1, 3, 256, 512)
        or meta.output_dtypes[embedding] != "float32"
    ):
        raise MetadataMismatchError("Embedding requires float32 [1,3,256,512]")
    if (
        meta.output_shapes[binary] not in ((1, 1, 256, 512), (1, 256, 512))
        or meta.output_dtypes[binary] != "int64"
    ):
        raise MetadataMismatchError(
            "Binary prediction requires int64 [1,1,256,512] or [1,256,512]"
        )
    return ModelBinding(selection, meta, inp, embedding, binary)


def create_runner(selection: ModelSelection, *, runtime_factory=None, runtime=None) -> NamedArrayRunner:
    """Construct the lazy LaneNet transport for a resolved selection.

    Args:
        selection: Manifest-backed selection carrying target, asset, and path.
        runtime_factory: Optional model-path-to-SDK-object factory (host seam).
        runtime: Optional prebuilt SDK object; overrides the factory.

    Returns:
        NamedArrayRunner: Lazy multi-output runner bound to the normalized
        RGB F32 physical input contract; loading gates board identity and
        the published file hash.
    """
    return NamedArrayRunner(
        selection,
        binding_loader=bind_model,
        physical_input=lambda binding: (
            binding.metadata.input_shapes[binding.input_name],
            "float32",
        ),
        task_name="LaneNet",
        runtime_factory=runtime_factory,
        runtime=runtime,
        execution_target_gate=require_execution_target,
    )


def validate_raw(outputs, binding):
    """Validate observed raw outputs without assuming order or auxiliary semantics.

    Args:
        outputs: Raw output mapping returned by infer.
        binding: Validated LaneNet binding supplying bound metadata.

    Raises:
        ValueError: Names, shapes, dtypes, or values differ from the binding.
    """
    meta = binding.metadata
    if not isinstance(outputs, Mapping) or set(outputs) != set(meta.output_names):
        raise ValueError("Output names differ from bound metadata")
    for name in meta.output_names:
        value = outputs[name]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != meta.output_shapes[name]
            or value.dtype != np.dtype(meta.output_dtypes[name])
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Invalid raw output shape/type/values: {name}")


@dataclass(frozen=True)
class LaneResult:
    """Owned embedding and binary lane labels at model-grid resolution.

    Attributes:
        embedding: float32 CHW instance-embedding array.
        binary: uint8 0/1 lane mask shaped (256, 512).
    """

    embedding: np.ndarray
    binary: np.ndarray


@dataclass(frozen=True)
class LanePredictionDetails:
    """One predict call's owned result plus its prepared input and raw outputs.

    Callers that archive ``raw_outputs.npz`` request this record with
    ``return_details=True`` instead of recomputing stages.  It describes only
    its own call; the task never retains a last output.

    Attributes:
        result: Owned LaneResult of this call.
        prepared: Prepared input mapping of this call.
        raw: Raw output mapping of this call.
    """

    result: LaneResult
    prepared: dict
    raw: dict


class LaneNetSegmenter:
    """Segment lane pixels with a compiled LaneNet model.

    ``predict`` composes ``preprocess`` → ``infer`` → ``postprocess``; the
    established ``pre_process``/``forward``/``post_process`` names stay thin
    aliases of those implementations.

    Attributes:
        runner (NamedArrayRunner): Lazy shared multi-output transport.
        binding (ModelBinding): Validated input and output roles.
    """

    def __init__(self, selection: ModelSelection, *, runner=None):
        """Load the compiled model and validate its tensor protocol.

        Args:
            selection: Manifest-backed selection from ``cli.resolve_selection``.
            runner: Optional injected transport (host-test seam); defaults to
                the shared lazy runner with the published-file hash gate.

        Returns:
            None.

        Raises:
            ValueError: The selection or its local file is invalid.
            MetadataMismatchError: Runtime metadata violates the contract.
            RuntimeError: Board identity or SDK loading fails.
        """
        self.runner = runner if runner is not None else create_runner(selection)
        self.binding = self.runner.load()

    def preprocess(self, image):
        """Nonempty BGR uint8 HWC → source RGB/ImageNet float32 NCHW.

        Args:
            image: uint8 BGR array shaped (H, W, 3); INTER_AREA stretch
                preserves the source input policy. There is no letterbox,
                per-frame geometry, normalization guess or model-grid resize.

        Returns:
            dict: Input-name mapping of one contiguous float32
            (1, 3, 256, 512) normalized tensor.

        Raises:
            ValueError: The image data is invalid.
        """
        return {self.binding.input_name: image_to_tensor(image)}

    def infer(self, tensors):
        """Return all named raw outputs, including observed auxiliaries, unchanged.

        Args:
            tensors: Input mapping returned by ``preprocess``.

        Returns:
            Mapping[str, np.ndarray]: Owned raw outputs keyed by bound name;
            no decoding, activation, or geometry restoration here.

        Raises:
            MetadataMismatchError: Input or output violates the binding.
            RuntimeError: SDK execution fails.
        """
        return self.runner(tensors)

    def postprocess(self, outputs):
        """Bound raw tensors → owned float32 CHW embedding and uint8 0/1 labels.

        Args:
            outputs: Raw output mapping returned by infer.

        Returns:
            LaneResult: Owned embedding and (256, 512) uint8 binary labels.
            Binary labels must already be discrete 0/1. No sigmoid, argmax,
            cluster, color scaling, original-size restoration or
            quantized-logit guess occurs.

        Raises:
            ValueError: Names, shapes, dtypes, values, or label set invalid.
        """
        validate_raw(outputs, self.binding)
        binary = outputs[self.binding.binary_name]
        if not np.isin(binary, (0, 1)).all():
            raise ValueError("Binary prediction must contain only labels 0 and 1")
        return LaneResult(
            outputs[self.binding.embedding_name][0].copy(),
            binary.reshape(256, 512).astype(np.uint8, copy=True),
        )

    def predict(self, image, *, return_details=False):
        """Execute the same three stages once; no rendering, timing or IO.

        Args:
            image: uint8 BGR array shaped (H, W, 3).
            return_details: Wrap the LaneResult with this call's prepared
                input and raw outputs for archival callers.

        Returns:
            LaneResult: Owned embedding and binary labels; see postprocess.

        Raises:
            ValueError: Image data or raw outputs are invalid.
            MetadataMismatchError: Tensor structure violates the binding.
            RuntimeError: SDK execution fails.
        """
        prepared = self.preprocess(image)
        raw = self.infer(prepared)
        result = self.postprocess(raw)
        if return_details:
            return LanePredictionDetails(result, prepared, raw)
        return result

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

    def pre_process(self, image):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image)

    def forward(self, tensors):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(tensors)

    def post_process(self, outputs):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs)

    def __call__(self, image):
        """Delegate to predict with the same input and error contract."""
        return self.predict(image)


__all__ = ["LaneNetSegmenter", "LanePredictionDetails", "LaneResult",
           "NamedArrayRunner", "create_runner", "image_to_tensor", "validate_raw"]
