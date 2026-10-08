# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""UNet semantic segmentation: load, preprocess, infer, postprocess, predict.

``UNetSegmenter`` owns the fixed 512x512 Pascal VOC contract end to end:
construction loads the model through the shared lazy transport, and each
``predict`` call runs preprocess -> infer -> postprocess visible in this file.
Catalog selection and presentation live in ``cli.py``; coloring lives in
``cli.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from utils.py_utils.image import bgr_to_nv12_planes
from utils.py_utils.platforms import require_execution_target
from utils.py_utils.quantization import apply_output_transform, validate_scale_quantization
from utils.py_utils.runtime_meta import MetadataMismatchError, RuntimeMetadata
from utils.py_utils.single_array_runner import SingleArrayRunner

from samples.vision.unet.runtime.python.cli import ModelSelection

INPUT_SIZE = 512
NUM_CLASSES = 21


@dataclass(frozen=True)
class ModelBinding:
    """Validated UNet tensor protocol; fixed 512x512 Pascal VOC geometry.

    Attributes:
        selection: The manifest-backed selection this binding was built from.
        metadata: Board-observed model metadata validated against the contract.
        input_name: Bound NV12 input tensor name.
        output_name: Bound 21-class logits output tensor name.
    """

    selection: ModelSelection
    metadata: RuntimeMetadata
    input_name: str
    output_name: str

    @property
    def model_name(self) -> str:
        """Return the single submodel name declared by the artifact."""
        return self.metadata.model_name


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    """Validate the fixed source UNet tensor protocol.

    Args:
        selection: Manifest-backed selection whose asset and path must match.
        metadata: Board-observed metadata mapping or ``RuntimeMetadata``.

    Returns:
        ModelBinding: Validated tensor names for the 512x512 NV12 contract.

    Raises:
        BindingError: The selection does not match the exact manifest asset.
        MetadataMismatchError: Model, tensor, geometry, or dtype contract fails.
    """

    # Re-resolve caller-created selections: identity and path stay inseparable.
    from samples.vision.unet.runtime.python.cli import BindingError, resolve_selection

    resolved = resolve_selection(
        selection.target,
        variant=selection.variant,
        asset_id=selection.asset.reference,
        model_path=selection.model_path if selection.explicit_model_path else None,
    )
    if (
        selection.target != resolved.target
        or selection.asset != resolved.asset
        or Path(selection.model_path) != Path(resolved.model_path)
        or selection.explicit_model_path != resolved.explicit_model_path
    ):
        raise BindingError("ModelSelection does not match the exact manifest asset and path.")

    meta = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if meta.model_names != (meta.model_name,):
        raise MetadataMismatchError("UNet artifact must expose exactly one model.")
    if len(meta.input_names) != 1 or len(meta.output_names) != 1:
        raise MetadataMismatchError("UNet requires exactly one input and one output.")
    input_name, output_name = meta.input_names[0], meta.output_names[0]
    shape = meta.input_shapes.get(input_name, ())
    if shape not in ((1, 3, 512, 512), (1, 512, 512, 3), (1, 768, 512, 1)):
        raise MetadataMismatchError("UNet NV12 input metadata must describe 512x512 geometry.")
    if meta.input_dtypes.get(input_name) != "nv12":
        raise MetadataMismatchError("UNet requires an NV12 input, not a float featuremap.")
    output_shape = meta.output_shapes.get(output_name, ())
    if output_shape not in ((1, 21, 512, 512), (1, 512, 512, 21)):
        raise MetadataMismatchError("UNet output must be NCHW/NHWC logits for 21 classes at 512x512.")
    dtype = meta.output_dtypes.get(output_name)
    if dtype not in ("float32", "int8", "uint8", "int16", "int32"):
        raise MetadataMismatchError(f"Unsupported UNet output dtype {dtype!r}.")
    if dtype != "float32":
        validate_scale_quantization(meta.output_quants.get(output_name), output_shape)
    return ModelBinding(selection, meta, input_name, output_name)


def create_runner(selection: ModelSelection, *, runtime_factory=None, runtime=None) -> SingleArrayRunner:
    """Construct the lazy UNet transport for a resolved selection.

    Args:
        selection: Manifest-backed selection carrying target, asset, and path.
        runtime_factory: Optional model-path-to-SDK-object factory (host seam).
        runtime: Optional prebuilt SDK object; overrides the factory.

    Returns:
        SingleArrayRunner: Lazy runner bound to the UNet physical input
        contract; loading gates board identity and the published file hash.
    """

    return SingleArrayRunner(
        selection,
        binding_loader=bind_model,
        physical_input=lambda binding: ((1, 768, 512, 1), "uint8"),
        task_name="UNet",
        runtime_factory=runtime_factory,
        runtime=runtime,
        execution_target_gate=require_execution_target,
    )


@dataclass(frozen=True)
class ImageContext:
    """Original-image geometry for one prepared call.

    Attributes:
        original_height: Source image height in pixels.
        original_width: Source image width in pixels.
    """

    original_height: int
    original_width: int


@dataclass(frozen=True)
class PreparedInput:
    """Packed NV12 tensors plus this call's geometry context.

    Attributes:
        tensors: Input-name to packed NV12 uint8 ``(1, 768, 512, 1)`` mapping.
        context: Frozen source geometry of this call only.
    """

    tensors: Mapping[str, np.ndarray]
    context: ImageContext


class UNetSegmenter:
    """Segment one image into 21 Pascal VOC classes with a compiled UNet.

    The constructor accepts a resolved selection; see __init__ for the
    injection seam. Construction loads the model immediately.

    Attributes:
        runner (SingleArrayRunner): Lazy shared transport used by infer.
        binding (ModelBinding): Validated tensor names and runtime metadata.
    """

    def __init__(self, selection: ModelSelection, *, runner=None) -> None:
        """Load the compiled model and validate its tensor protocol.

        Args:
            selection: Manifest-backed selection from ``cli.resolve_selection``.
            runner: Optional injected transport (host-test seam); defaults to
                the shared lazy runner with the published-file hash gate.

        Returns:
            None.

        Raises:
            ValueError: The selection or its local file is invalid.
            MetadataMismatchError: Runtime metadata violates the UNet contract.
            RuntimeError: Board identity or SDK loading fails.
        """
        self.runner = runner if runner is not None else create_runner(selection)
        self.binding = self.runner.load()

    def preprocess(self, image: np.ndarray) -> PreparedInput:
        """Convert one BGR image to the model's packed NV12 input tensors.

        Args:
            image: uint8 BGR array shaped (H, W, 3) with values in [0, 255];
                not modified. Direct INTER_LINEAR resize preserves the source
                stretch geometry: no letterbox, normalization, or automatic
                original-resolution restoration.

        Returns:
            PreparedInput: Input-name mapping of contiguous uint8 packed NV12
            bytes shaped (1, 768, 512, 1), plus this call's frozen context.

        Raises:
            ValueError: The array shape, dtype, or dimensions are invalid.
        """
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
                or image.dtype != np.uint8 or not all(image.shape[:2])):
            raise ValueError('Expected a nonempty BGR HWC uint8 image.')
        resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        y, uv = bgr_to_nv12_planes(resized)
        packed = np.concatenate((y.reshape(-1), uv.reshape(-1))).reshape(1, 768, 512, 1)
        return PreparedInput({self.binding.input_name: packed}, ImageContext(*image.shape[:2]))

    def infer(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Execute one inference call with the prepared physical tensors.

        Args:
            tensors: Input mapping returned by ``preprocess(...).tensors``.

        Returns:
            np.ndarray: Raw owned logits squeezed from the single bound output,
            NCHW ``(1, 21, 512, 512)`` or NHWC ``(1, 512, 512, 21)``. float32
            stays raw; integer dtypes keep the SDK's raw values. No
            dequantization, activation, or decoding happens here.

        Raises:
            MetadataMismatchError: Input names, shapes, dtype, or the runtime
                output structure violate the binding.
            RuntimeError: SDK execution fails.
        """
        return self.runner(tensors)

    def postprocess(self, raw: np.ndarray) -> np.ndarray:
        """Decode raw 21-class logits into a model-resolution class mask.

        Args:
            raw: Output of ``infer``. Integer logits are SCALE-dequantized;
            float32 is already decoded (a vestigial descriptor is not applied).

        Returns:
            np.ndarray: Owned uint8 ``(512, 512)`` class IDs in [0, 20].
            Argmax ties select the lowest class ID. The source contract
            returns model-resolution masks, so no context restoration occurs.

        Raises:
            ValueError: Shape, dtype, nonfinite values, or dequantized scores
                are invalid.
        """
        name = self.binding.output_name
        meta = self.binding.metadata
        if (not isinstance(raw, np.ndarray) or raw.shape != meta.output_shapes[name]
                or raw.dtype != np.dtype(meta.output_dtypes[name]) or not np.isfinite(raw).all()):
            raise ValueError('UNet raw output does not match bound shape/dtype/finite values.')
        transform = 'raw_f32' if raw.dtype == np.float32 else 'dequant'
        scores = apply_output_transform(transform, {name: raw}, meta.output_quants)[name]
        if not np.isfinite(scores).all():
            raise ValueError('UNet dequantization produced nonfinite scores.')
        axis = 0 if scores.shape[1] == NUM_CLASSES else -1
        return scores[0].argmax(axis=axis).astype(np.uint8)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run preprocessing, inference, and postprocessing for one image.

        Args:
            image: uint8 BGR array shaped (H, W, 3), values [0, 255].

        Returns:
            np.ndarray: uint8 ``(512, 512)`` class-ID mask; see postprocess.

        Raises:
            ValueError: Image data, tensors, or decoded scores are invalid.
            MetadataMismatchError: Tensor structure violates the binding.
            RuntimeError: SDK execution fails.
        """
        prepared = self.preprocess(image)
        return self.postprocess(self.infer(prepared.tensors))

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

    def pre_process(self, image: np.ndarray) -> PreparedInput:
        """Delegate to preprocess with the same input and error contract."""
        return self.preprocess(image)

    def forward(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Delegate to infer with the same input and error contract."""
        return self.infer(tensors)

    def post_process(self, raw: np.ndarray) -> np.ndarray:
        """Delegate to postprocess with the same input and error contract."""
        return self.postprocess(raw)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Delegate to predict with the same input and error contract."""
        return self.predict(image)
