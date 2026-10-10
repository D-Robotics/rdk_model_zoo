# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Depth Anything V2 depth estimation: load, preprocess, infer, postprocess, predict.

``DepthEstimator`` owns the fixed 518x686 RGB contract end to end:
construction loads the model through the shared lazy transport, and each
``predict`` call runs preprocess -> infer -> postprocess visible in this
file, together with the per-frame resize/letterbox geometry it consumes.
Catalog selection and presentation live in ``cli.py``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import cv2
import numpy as np
from utils.py_utils.platforms import require_execution_target
from utils.py_utils.runtime_meta import MetadataMismatchError, RuntimeMetadata
from utils.py_utils.single_array_runner import SingleArrayRunner
from samples.vision.depth_anything_v2.runtime.python.cli import ModelSelection

INPUT_HEIGHT = 518
INPUT_WIDTH = 686


# ---------------------------------------------------------------------
# Per-frame source resize geometry; no mutable task state.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ImageContext:
    original_h: int
    original_w: int
    resize_type: int
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0


def make_context(height, width, resize_type):
    if type(height) is not int or type(width) is not int or min(height, width) <= 0:
        raise ValueError("Image dimensions must be positive integers")
    if type(resize_type) is not int or resize_type not in (0, 1):
        raise ValueError("resize_type must be 0 (stretch) or 1 (letterbox)")
    if resize_type == 0:
        return ImageContext(height, width, resize_type)
    scale = min(518 / height, 686 / width)
    h, w = int(height * scale), int(width * scale)
    if min(h, w) <= 0:
        raise ValueError("Letterbox dimension collapsed to zero")
    ph, pw = 518 - h, 686 - w
    return ImageContext(
        height, width, resize_type, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2
    )


def validate_context(context, resize_type):
    if not isinstance(context, ImageContext) or context.resize_type != resize_type:
        raise ValueError("Wrong image context/profile")
    if context != make_context(context.original_h, context.original_w, resize_type):
        raise ValueError("Geometry context does not match original dimensions")


@dataclass(frozen=True)
class ModelBinding:
    """Validated fixed RGB featuremap and relative-depth tensor protocol.

    Attributes:
        selection: The manifest-backed selection this binding was built from.
        metadata: Board-observed model metadata validated against the contract.
        input_name: Bound normalized RGB F32 input tensor name.
        output_name: Bound relative-depth F32 output tensor name.
    """

    selection: ModelSelection
    metadata: RuntimeMetadata
    input_name: str
    output_name: str

    @property
    def model_name(self) -> str:
        """Return the single submodel name declared by the artifact."""
        return self.metadata.model_name


def bind_model(
    selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]
) -> ModelBinding:
    """Bind the source 518×686 RGB featuremap and relative-depth output.

    Args:
        selection: Manifest-backed selection whose asset and path must match.
        metadata: Board-observed metadata mapping or ``RuntimeMetadata``.

    Returns:
        ModelBinding: Validated tensor names for the fixed float32 contract.

    Raises:
        BindingError: The selection differs from the exact published asset.
        MetadataMismatchError: Tensor names, shapes, or dtypes violate the
            contract.

    Notes:
        Internal int16 quantization in the source guide is not evidence that
        the public output tensor is int16. Accept only the float32 source IO
        contract.
    """
    from samples.vision.depth_anything_v2.runtime.python.cli import resolve_selection

    expected = resolve_selection(
        selection.target,
        asset_id=selection.asset.reference,
        model_path=selection.model_path if selection.explicit_model_path else None,
    )
    if expected != selection:
        from samples.vision.depth_anything_v2.runtime.python.cli import BindingError
        raise BindingError("Selection differs from the exact published asset")
    meta = (
        metadata
        if isinstance(metadata, RuntimeMetadata)
        else RuntimeMetadata.from_mapping(metadata)
    )
    if (
        len(meta.model_names) != 1
        or len(meta.input_names) != 1
        or len(meta.output_names) != 1
    ):
        raise MetadataMismatchError("Expected one model, one input and one output")
    inp, out = meta.input_names[0], meta.output_names[0]
    if (
        meta.input_shapes.get(inp) != (1, 3, INPUT_HEIGHT, INPUT_WIDTH)
        or meta.input_dtypes.get(inp) != "float32"
    ):
        raise MetadataMismatchError("Expected float32 RGB NCHW [1,3,518,686]")
    if (
        meta.output_shapes.get(out) != (1, INPUT_HEIGHT, INPUT_WIDTH)
        or meta.output_dtypes.get(out) != "float32"
    ):
        raise MetadataMismatchError("Expected float32 depth [1,518,686]")
    return ModelBinding(selection, meta, inp, out)


def create_runner(selection: ModelSelection, *, runtime_factory=None, runtime=None) -> SingleArrayRunner:
    """Construct the lazy Depth Anything V2 transport for a resolved selection.

    Args:
        selection: Manifest-backed selection carrying target, asset, and path.
        runtime_factory: Optional model-path-to-SDK-object factory (host seam).
        runtime: Optional prebuilt SDK object; overrides the factory.

    Returns:
        SingleArrayRunner: Lazy runner bound to the normalized RGB F32
        physical input contract; loading gates board identity and the
        published file hash.
    """
    return SingleArrayRunner(
        selection,
        binding_loader=bind_model,
        physical_input=lambda binding: (
            binding.metadata.input_shapes[binding.input_name],
            "float32",
        ),
        task_name="Depth Anything V2",
        runtime_factory=runtime_factory,
        runtime=runtime,
        execution_target_gate=require_execution_target,
    )


@dataclass(frozen=True)
class PreparedInput:
    tensors: Mapping[str, np.ndarray]
    context: ImageContext


@dataclass(frozen=True)
class DepthResult:
    depth_native: np.ndarray
    context: ImageContext


@dataclass(frozen=True)
class DepthPredictionDetails:
    """One predict call's owned result plus its prepared input and raw output.

    Callers that archive the raw tensor (``raw_depth.npy``) request this record
    with ``return_details=True`` instead of recomputing stages.  It describes
    only its own call; the task never retains a last image or last output.
    """

    result: DepthResult
    prepared: PreparedInput
    raw: np.ndarray


class DepthEstimator:
    """Estimate one image's relative depth with a compiled Depth Anything V2.

    ``predict`` composes ``preprocess`` → ``infer`` → ``postprocess``; the
    established ``pre_process``/``forward``/``post_process`` names stay thin
    aliases of those implementations.

    Attributes:
        runner (SingleArrayRunner): Lazy shared transport used by infer.
        binding (ModelBinding): Validated tensor names and runtime metadata.
        resize_type: 0 stretch (default) or 1 letterbox preprocessing.
    """

    def __init__(self, selection: ModelSelection, *, resize_type=0, runner=None):
        """Load the compiled model and validate its tensor protocol.

        Args:
            selection: Manifest-backed selection from ``cli.resolve_selection``.
            resize_type: 0 stretch or 1 letterbox; validated eagerly.
            runner: Optional injected transport (host-test seam); defaults to
                the shared lazy runner with the published-file hash gate.

        Returns:
            None.

        Raises:
            ValueError: The resize_type or selection is invalid.
            MetadataMismatchError: Runtime metadata violates the contract.
            RuntimeError: Board identity or SDK loading fails.
        """
        make_context(1, 1, resize_type)
        self.runner = runner if runner is not None else create_runner(selection)
        self.binding = self.runner.load()
        self.resize_type = resize_type

    # ------------------------------------------------------------------
    # The three pipeline stages, each public and usable on its own.
    # ------------------------------------------------------------------

    def preprocess(self, image):
        """BGR uint8 HWC → owned normalized RGB float32 NCHW plus geometry.

        Stretch uses source INTER_NEAREST; optional letterbox uses INTER_LINEAR
        and gray127. Normalize each pixel across RGB, not ImageNet constants.
        """
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or min(image.shape[:2]) <= 0
        ):
            raise ValueError("Expected nonempty BGR uint8 HWC image")
        ctx = make_context(*image.shape[:2], self.resize_type)
        if self.resize_type == 0:
            resized = cv2.resize(image, (686, 518), interpolation=cv2.INTER_NEAREST)
        else:
            resized = cv2.resize(
                image,
                (686 - ctx.left - ctx.right, 518 - ctx.top - ctx.bottom),
                interpolation=cv2.INTER_LINEAR,
            )
            resized = cv2.copyMakeBorder(
                resized,
                ctx.top,
                ctx.bottom,
                ctx.left,
                ctx.right,
                cv2.BORDER_CONSTANT,
                value=(127, 127, 127),
            )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb - rgb.mean(axis=-1, keepdims=True)) / np.sqrt(
            rgb.var(axis=-1, keepdims=True) + 1e-5
        )
        tensor = np.ascontiguousarray(
            normalized.transpose(2, 0, 1)[None], dtype=np.float32
        )
        return PreparedInput({self.binding.input_name: tensor}, ctx)

    def infer(self, tensors):
        """Return owned raw float32 [1,518,686] without scaling or rendering."""
        return self.runner(tensors)

    def postprocess(self, raw, context):
        """Finite float depth → crop optional padding, restore original H×W.

        OpenCV bilinear replaces source Torch align_corners=False. No numerical
        bit-identity is claimed. Output retains relative values, not meters or
        display-normalized intensities; display normalization lives in ``cli.py``.
        """
        validate_context(context, self.resize_type)
        if (
            not isinstance(raw, np.ndarray)
            or raw.shape != (1, 518, 686)
            or raw.dtype != np.float32
            or not np.isfinite(raw).all()
        ):
            raise ValueError("Expected finite float32 [1,518,686] output")
        plane = raw[
            0, context.top : 518 - context.bottom, context.left : 686 - context.right
        ]
        result = cv2.resize(
            plane,
            (context.original_w, context.original_h),
            interpolation=cv2.INTER_LINEAR,
        )
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite restored depth")
        return DepthResult(result.copy(), context)

    def predict(self, image, *, return_details=False):
        """Execute the same three stages with no IO, visualization or timing.

        ``return_details=True`` wraps the usual :class:`DepthResult` with this
        call's prepared input and raw output, so one production inference also
        serves callers that archive ``raw_depth.npy``; the default return stays
        the plain :class:`DepthResult`.
        """
        prepared = self.preprocess(image)
        raw = self.infer(prepared.tensors)
        result = self.postprocess(raw, prepared.context)
        if return_details:
            return DepthPredictionDetails(result, prepared, raw)
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

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin aliases
    # of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, image):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image)

    def forward(self, tensors):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(tensors)

    def post_process(self, raw, context):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(raw, context)
