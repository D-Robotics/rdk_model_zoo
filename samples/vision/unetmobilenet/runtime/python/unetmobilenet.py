# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""UnetMobileNet Cityscapes segmentation: load, preprocess, infer, postprocess, predict.

``UnetMobileNetSegmenter`` owns the fixed 2048x1024 split-NV12 contract end to
end: construction loads the model through the shared lazy transport, and each
``predict`` call runs preprocess -> infer -> postprocess visible in this file.
Catalog selection and presentation live in ``cli.py``; coloring lives in
``cli.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from utils.py_utils.image import bgr_to_nv12_planes
from utils.py_utils.platforms import require_execution_target
from utils.py_utils.quantization import dequantize_tensor, validate_scale_quantization
from utils.py_utils.runtime_meta import MetadataMismatchError, RuntimeMetadata
from utils.py_utils.single_array_runner import SingleArrayRunner

from samples.vision.unetmobilenet.runtime.python.cli import ModelSelection

INPUT_HEIGHT = 1024
INPUT_WIDTH = 2048
NUM_CLASSES = 19


@dataclass(frozen=True)
class ModelBinding:
    """Validated split-NV12 UnetMobileNet tensor protocol.

    Attributes:
        selection: The manifest-backed selection this binding was built from.
        metadata: Board-observed model metadata validated against the contract.
        y_name: Bound luma plane input name.
        uv_name: Bound chroma plane input name.
        output_name: Bound NHWC 19-class logits output name.
        input_height: Fixed model input height (1024).
        input_width: Fixed model input width (2048).
    """

    selection: ModelSelection
    metadata: RuntimeMetadata
    y_name: str
    uv_name: str
    output_name: str
    input_height: int = INPUT_HEIGHT
    input_width: int = INPUT_WIDTH

    @property
    def model_name(self) -> str:
        """Return the single submodel name declared by the artifact."""
        return self.metadata.model_name


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    """Validate the source UnetMobileNet split-NV12 tensor protocol.

    Args:
        selection: Manifest-backed selection whose asset and path must match.
        metadata: Board-observed metadata mapping or ``RuntimeMetadata``.

    Returns:
        ModelBinding: Validated Y/UV input names and NHWC logits output.

    Raises:
        ValueError: The selection does not match the manifest identity.
        MetadataMismatchError: Model, tensor, geometry, or dtype contract fails.
    """
    from samples.vision.unetmobilenet.runtime.python.cli import resolve_selection

    resolved = resolve_selection(selection.target, asset_id=selection.asset.reference,
                                 model_path=selection.model_path if selection.explicit_model_path else None)
    if selection != resolved:
        raise ValueError('ModelSelection does not match the manifest identity and path')
    meta = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if meta.model_names != (meta.model_name,) or len(meta.input_names) != 2 or len(meta.output_names) != 1:
        raise MetadataMismatchError('UnetMobileNet requires one model, split Y/UV and one logits tensor')
    y = [name for name in meta.input_names if meta.input_shapes.get(name) == (1, INPUT_HEIGHT, INPUT_WIDTH, 1)]
    uv = [name for name in meta.input_names if meta.input_shapes.get(name) == (1, INPUT_HEIGHT // 2, INPUT_WIDTH // 2, 2)]
    if len(y) != 1 or len(uv) != 1 or any(meta.input_dtypes.get(name) != 'uint8' for name in meta.input_names):
        raise MetadataMismatchError('Expected uint8 Y [1,1024,2048,1] and UV [1,512,1024,2]')
    output = meta.output_names[0]
    shape = meta.output_shapes.get(output, ())
    if len(shape) != 4 or shape[0] != 1 or shape[-1] != NUM_CLASSES or min(shape[1:3]) <= 0:
        raise MetadataMismatchError('Expected NHWC [1,H,W,19] segmentation logits')
    dtype = meta.output_dtypes.get(output)
    if dtype not in ('int32', 'float32'):
        raise MetadataMismatchError('Expected source int32 or explicit float32 logits')
    if dtype == 'int32':
        info = meta.output_quants.get(output)
        kind = getattr(info, 'quant_type', None)
        kind = str(getattr(kind, 'name', kind))
        if kind in ('SCALE', '1'):
            if not hasattr(info, 'zero_point'):
                raise MetadataMismatchError('SCALE requires zero_point metadata (empty means symmetric)')
            validate_scale_quantization(info, shape)
        elif kind not in ('NONE', '0'):
            raise MetadataMismatchError('int32 logits require explicit NONE or valid SCALE metadata')
    return ModelBinding(selection, meta, y[0], uv[0], output)


def create_runner(selection: ModelSelection, *, runtime_factory=None, runtime=None) -> SingleArrayRunner:
    """Construct the lazy UnetMobileNet transport for a resolved selection.

    Args:
        selection: Manifest-backed selection carrying target, asset, and path.
        runtime_factory: Optional model-path-to-SDK-object factory (host seam).
        runtime: Optional prebuilt SDK object; overrides the factory.

    Returns:
        SingleArrayRunner: Lazy runner bound to the split Y/UV physical input
        contract; loading gates board identity and the published file hash.
    """
    return SingleArrayRunner(
        selection,
        binding_loader=bind_model,
        physical_inputs=lambda binding: {
            binding.y_name: ((1, INPUT_HEIGHT, INPUT_WIDTH, 1), 'uint8'),
            binding.uv_name: ((1, INPUT_HEIGHT // 2, INPUT_WIDTH // 2, 2), 'uint8'),
        },
        task_name='UnetMobileNet',
        runtime=runtime,
        runtime_factory=runtime_factory,
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
    """Split-NV12 tensors plus this call's geometry context.

    Attributes:
        tensors: Y/UV-name to contiguous uint8 plane mapping.
        context: Frozen source geometry of this call only.
    """

    tensors: Mapping[str, np.ndarray]
    context: ImageContext


class UnetMobileNetSegmenter:
    """Segment one image into 19 Cityscapes classes with a compiled model.

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
            MetadataMismatchError: Runtime metadata violates the contract.
            RuntimeError: Board identity or SDK loading fails.
        """
        self.runner = runner if runner is not None else create_runner(selection)
        self.binding = self.runner.load()

    def preprocess(self, image: np.ndarray) -> PreparedInput:
        """Convert one BGR image to the model's split NV12 input tensors.

        Args:
            image: uint8 BGR array shaped (H, W, 3), values [0, 255]; not
                modified. Stretched with INTER_AREA to 2048x1024; no
                normalization or letterbox.

        Returns:
            PreparedInput: Y plane (1, 1024, 2048, 1) and UV plane
            (1, 512, 1024, 2), contiguous uint8, plus this call's frozen
            original geometry.

        Raises:
            ValueError: The array shape, dtype, or dimensions are invalid.
        """
        if (not isinstance(image, np.ndarray) or image.ndim != 3
                or image.shape[2] != 3 or image.dtype != np.uint8
                or not all(image.shape[:2])):
            raise ValueError('Expected nonempty BGR uint8 HWC input')
        resized = cv2.resize(image, (self.binding.input_width, self.binding.input_height),
                             interpolation=cv2.INTER_AREA)
        y, uv = bgr_to_nv12_planes(resized)
        return PreparedInput({self.binding.y_name: y, self.binding.uv_name: uv},
                             ImageContext(*image.shape[:2]))

    def infer(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Execute one inference call with the prepared physical tensors.

        Args:
            tensors: Input mapping returned by ``preprocess(...).tensors``.

        Returns:
            np.ndarray: Raw owned NHWC logits (1, H, W, 19), int32 or float32,
            unchanged: no dequantization, activation, or decoding here.

        Raises:
            MetadataMismatchError: Input names, shapes, dtype, or the runtime
                output structure violate the binding.
            RuntimeError: SDK execution fails.
        """
        return self.runner(tensors)

    def postprocess(self, raw: np.ndarray, context: ImageContext) -> np.ndarray:
        """Decode scores and resize class IDs to the original image size.

        Args:
            raw: Output of ``infer``. int32 SCALE logits are affine-dequantized
                only here using float64 comparison to retain int32 differences;
                explicit NONE int32 and float32 remain raw.
            context: This call's ``ImageContext`` from preprocess.

        Returns:
            np.ndarray: Owned int32 mask (original_height, original_width) of
            class IDs 0..18. Argmax ties choose the lowest class ID;
            INTER_NEAREST restores geometry. No coloring or blending.

        Raises:
            ValueError: Logits or context shape/dtype/values are invalid.
        """
        meta = self.binding.metadata
        name = self.binding.output_name
        if (not isinstance(raw, np.ndarray) or raw.shape != meta.output_shapes[name]
                or raw.dtype != np.dtype(meta.output_dtypes[name]) or not np.isfinite(raw).all()):
            raise ValueError('Logits do not match the bound shape/dtype or contain nonfinite values')
        if (not isinstance(context, ImageContext) or context.original_height <= 0
                or context.original_width <= 0):
            raise ValueError('A valid per-call ImageContext is required')
        scores = raw
        if raw.dtype == np.int32:
            scores = dequantize_tensor(raw, meta.output_quants[name], dtype='float64')
        labels = np.argmax(scores[0], axis=-1).astype(np.int32)
        return cv2.resize(labels, (context.original_width, context.original_height),
                          interpolation=cv2.INTER_NEAREST).copy()

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run preprocessing, inference, and postprocessing for one image.

        Args:
            image: uint8 BGR array shaped (H, W, 3), values [0, 255].

        Returns:
            np.ndarray: int32 class-ID mask at the original image size; see
            postprocess.

        Raises:
            ValueError: Image data, tensors, or context are invalid.
            MetadataMismatchError: Tensor structure violates the binding.
            RuntimeError: SDK execution fails.
        """
        prepared = self.preprocess(image)
        return self.postprocess(self.infer(prepared.tensors), prepared.context)

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

    def post_process(self, raw: np.ndarray, context: ImageContext) -> np.ndarray:
        """Delegate to postprocess with the same input and error contract."""
        return self.postprocess(raw, context)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Delegate to predict with the same input and error contract."""
        return self.predict(image)
