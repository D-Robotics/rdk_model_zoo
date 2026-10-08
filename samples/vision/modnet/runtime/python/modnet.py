# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""MODNet portrait matting: load, preprocess, infer, postprocess, predict.

``MODNetMatting`` owns the fixed 512x512 RGB float contract end to end:
construction loads the model through the shared lazy transport, and each
``predict`` call runs preprocess -> infer -> postprocess visible in this
file. Catalog selection and presentation live in ``cli.py``; compositing
lives in ``cli.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from utils.py_utils.platforms import require_execution_target
from utils.py_utils.runtime_meta import MetadataMismatchError, RuntimeMetadata
from utils.py_utils.single_array_runner import SingleArrayRunner

from samples.vision.modnet.runtime.python.cli import ModelSelection

INPUT_SHAPE = (1, 3, 512, 512)
OUTPUT_SHAPE = (1, 1, 512, 512)
REF_SIZE = 512


@dataclass(frozen=True)
class ModelBinding:
    """Validated MODNet input/output names and source-proven shapes.

    Attributes:
        selection: The manifest-backed selection this binding was built from.
        metadata: Board-observed model metadata validated against the contract.
        input_name: Bound RGB F32 NCHW input tensor name.
        output_name: Bound F32 matte output tensor name.
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
    """Validate the fixed source MODNet tensor protocol.

    Args:
        selection: Manifest-backed selection whose asset and path must match.
        metadata: Board-observed metadata mapping or ``RuntimeMetadata``.

    Returns:
        ModelBinding: Validated tensor names for the 512x512 F32 contract.

    Raises:
        BindingError: The selection does not match the exact manifest asset.
        MetadataMismatchError: Model, tensor, geometry, or dtype contract fails.
    """
    from samples.vision.modnet.runtime.python.cli import BindingError, resolve_selection

    # Re-resolve before accepting a caller-created selection.  MODNet's asset is
    # manual, so its manifest identity and explicit path must remain inseparable.
    resolved = resolve_selection(
        selection.target,
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
        raise MetadataMismatchError("MODNet artifact must expose exactly one model.")
    if len(meta.input_names) != 1 or len(meta.output_names) != 1:
        raise MetadataMismatchError("MODNet requires exactly one input and one output.")
    input_name, output_name = meta.input_names[0], meta.output_names[0]
    if meta.input_shapes.get(input_name) != INPUT_SHAPE:
        raise MetadataMismatchError(f"Expected input shape {INPUT_SHAPE}, got {meta.input_shapes.get(input_name)}.")
    if meta.input_dtypes.get(input_name) != "float32":
        raise MetadataMismatchError("MODNet input must be float32.")
    if meta.output_shapes.get(output_name) != OUTPUT_SHAPE:
        raise MetadataMismatchError(f"Expected output shape {OUTPUT_SHAPE}, got {meta.output_shapes.get(output_name)}.")
    if meta.output_dtypes.get(output_name) != "float32":
        raise MetadataMismatchError("MODNet output must be float32.")
    return ModelBinding(selection, meta, input_name, output_name)


def create_runner(selection: ModelSelection, *, runtime_factory=None, runtime=None) -> SingleArrayRunner:
    """Construct the lazy MODNet transport for a resolved selection.

    Args:
        selection: Manifest-backed selection carrying target, asset, and path.
        runtime_factory: Optional model-path-to-SDK-object factory (host seam).
        runtime: Optional prebuilt SDK object; overrides the factory.

    Returns:
        SingleArrayRunner: Lazy runner bound to the RGB F32 NCHW physical
        input contract; loading gates board identity and the published file.
    """
    return SingleArrayRunner(
        selection,
        binding_loader=bind_model,
        physical_input=lambda binding: (INPUT_SHAPE, "float32"),
        task_name="MODNet",
        runtime_factory=runtime_factory,
        runtime=runtime,
        execution_target_gate=require_execution_target,
    )


@dataclass(frozen=True)
class GeometryContext:
    """Immutable geometry required to restore one matte to one source image.

    Attributes:
        original_height: Source image height in pixels.
        original_width: Source image width in pixels.
        pad_x: Left zero-padding width applied after resize.
        pad_y: Top zero-padding height applied after resize.
        resized_width: Resized content width before padding.
        resized_height: Resized content height before padding.
        target_size: Square model input size (512).
    """

    original_height: int
    original_width: int
    pad_x: int
    pad_y: int
    resized_width: int
    resized_height: int
    target_size: int


@dataclass(frozen=True)
class PreparedInput:
    """One MODNet tensor mapping and its independent geometry context.

    Attributes:
        tensors: Input-name to F32 NCHW tensor mapping.
        context: Frozen restore geometry of this call only.
    """

    tensors: Mapping[str, np.ndarray]
    context: GeometryContext


def resize_with_padding(image: np.ndarray, target_size: int) -> tuple[np.ndarray, GeometryContext]:
    """Apply the source long-side resize, centered zero padding, and context capture.

    Args:
        image: HWC image array with any float dtype.
        target_size: Square output size in pixels.

    Returns:
        tuple: Padded (target_size, target_size, C) array and the frozen
        GeometryContext describing this call's resize and padding.

    Raises:
        ValueError: The image is not HWC or a dimension is nonpositive.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR HWC image with three channels.")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0 or target_size <= 0:
        raise ValueError("Image and target size must be positive.")
    scale = target_size / max(height, width)
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    pad_w = target_size - resized_width
    pad_h = target_size - resized_height
    pad_x = pad_w // 2
    pad_y = pad_h // 2
    padded = cv2.copyMakeBorder(
        resized, pad_y, pad_h - pad_y, pad_x, pad_w - pad_x,
        cv2.BORDER_CONSTANT, value=0,
    )
    return padded, GeometryContext(height, width, pad_x, pad_y, resized_width, resized_height, target_size)


class MODNetMatting:
    """Extract one portrait matte with a compiled MODNet model.

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
                the shared lazy runner with the board-identity and file gates.

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
        """Convert BGR HWC input to source RGB F32 NCHW ``[-1,1]``.

        Args:
            image: uint8 BGR array shaped (H, W, 3); not modified. Converted to
                RGB, normalized to [-1, 1], long-side resized with INTER_AREA,
                and center-padded with zeros to 512x512.

        Returns:
            PreparedInput: Input-name mapping of one contiguous float32
            (1, 3, 512, 512) tensor, plus this call's frozen geometry.

        Raises:
            ValueError: The image is None or not an HWC three-channel array.
        """
        if image is None:
            raise ValueError("Input image is None.")
        rgb = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) - 127.5) / 127.5
        padded, context = resize_with_padding(normalized, REF_SIZE)
        tensor = np.transpose(padded, (2, 0, 1))[None].astype(np.float32, copy=True)
        return PreparedInput({self.binding.input_name: tensor}, context)

    def infer(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Execute one inference call with the prepared physical tensors.

        Args:
            tensors: Input mapping returned by ``preprocess(...).tensors``.

        Returns:
            np.ndarray: Raw owned float32 matte shaped (1, 1, 512, 512) with
            values in [0, 1]; no thresholding or geometry restoration here.

        Raises:
            MetadataMismatchError: Input names, shapes, dtype, or the runtime
                output structure violate the binding.
            RuntimeError: SDK execution fails.
        """
        return self.runner(tensors)

    def postprocess(self, raw: np.ndarray, context: GeometryContext) -> np.ndarray:
        """Convert a raw ``[0,1]`` matte to uint8 original-image geometry.

        Args:
            raw: Output of ``infer``. Scaled by 255, cropped back to the
                resized content window, and restored to the source size.
            context: This call's ``GeometryContext`` from preprocess.

        Returns:
            np.ndarray: uint8 matte shaped (original_height, original_width).

        Raises:
            ValueError: Shape or dtype does not match the bound contract.
        """
        value = np.asarray(raw)
        if value.shape != (1, 1, context.target_size, context.target_size) or value.dtype != np.float32:
            raise ValueError(f"Expected raw float32 matte (1,1,{context.target_size},{context.target_size}), got {value.shape}/{value.dtype}.")
        matte = (value[0, 0] * 255).astype(np.uint8)
        unpadded = matte[
            context.pad_y:context.pad_y + context.resized_height,
            context.pad_x:context.pad_x + context.resized_width,
        ]
        return cv2.resize(unpadded, (context.original_width, context.original_height), interpolation=cv2.INTER_LINEAR)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run preprocessing, inference, and postprocessing for one image.

        Args:
            image: uint8 BGR array shaped (H, W, 3).

        Returns:
            np.ndarray: uint8 matte at the original image size; see postprocess.

        Raises:
            ValueError: Image data, tensors, or geometry context are invalid.
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

    def post_process(self, raw: np.ndarray, context: GeometryContext) -> np.ndarray:
        """Delegate to postprocess with the same input and error contract."""
        return self.postprocess(raw, context)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Delegate to predict with the same input and error contract."""
        return self.predict(image)


__all__ = ["GeometryContext", "MODNetMatting", "PreparedInput", "resize_with_padding"]
