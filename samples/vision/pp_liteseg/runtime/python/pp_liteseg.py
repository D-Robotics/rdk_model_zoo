# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""PP-LiteSeg semantic segmentation: load, preprocess, infer, postprocess, predict.

``PPLiteSegSegmenter`` owns the fixed 1024x512 Cityscapes segmentation contract
end to end: construction loads the model through the shared lazy transport,
and each ``predict`` call runs preprocess -> infer -> postprocess visible in
this file. Catalog selection and presentation live in ``cli.py``; rendering
lives in ``cli.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from utils.py_utils.image import bgr_to_nv12_planes
from utils.py_utils.platforms import require_execution_target
from utils.py_utils.runtime_meta import MetadataMismatchError, RuntimeMetadata
from utils.py_utils.single_array_runner import SingleArrayRunner

from samples.vision.pp_liteseg.runtime.python.cli import ModelSelection

INPUT_HEIGHT = 512
INPUT_WIDTH = 1024
NUM_CLASSES = 19


@dataclass(frozen=True)
class ModelBinding:
    """Validated X5 class-map or separately registered S float-logits protocol.

    Attributes:
        selection: The manifest-backed selection this binding was built from.
        metadata: Board-observed model metadata validated against the contract.
        input_name: Bound input tensor name.
        output_name: Bound int32 class-map or float-logits output tensor name.
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
    """Validate the selected artifact's exact input and output protocol.

    Args:
        selection: Manifest-backed selection whose asset and path must match.
        metadata: Board-observed metadata mapping or ``RuntimeMetadata``.

    Returns:
        ModelBinding: Validated tensor names for the 1024x512 NV12 contract.

    Raises:
        BindingError: The selection does not match the exact manifest asset.
        MetadataMismatchError: Model, tensor, geometry, or dtype contract fails.
    """
    from samples.vision.pp_liteseg.runtime.python.cli import BindingError, resolve_selection

    # Re-resolve caller-created selections: identity and path stay inseparable.
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
        raise MetadataMismatchError("PP-LiteSeg artifact must expose exactly one model.")
    if len(meta.input_names) != 1 or len(meta.output_names) != 1:
        raise MetadataMismatchError("PP-LiteSeg requires exactly one input and one output.")
    input_name, output_name = meta.input_names[0], meta.output_names[0]
    if selection.float_logits:
        if meta.input_shapes.get(input_name)!=(1,3,INPUT_HEIGHT,INPUT_WIDTH) or meta.input_dtypes.get(input_name)!='float32':
            raise MetadataMismatchError('S PP-LiteSeg requires normalized RGB float32 NCHW (1,3,512,1024).')
        if meta.output_shapes.get(output_name) not in ((1,NUM_CLASSES,INPUT_HEIGHT,INPUT_WIDTH),(1,INPUT_HEIGHT,INPUT_WIDTH,NUM_CLASSES)) or meta.output_dtypes.get(output_name)!='float32':
            raise MetadataMismatchError('S PP-LiteSeg requires a float32 19-class logits tensor.')
        return ModelBinding(selection,meta,input_name,output_name)
    shape = meta.input_shapes.get(input_name, ())
    if shape not in ((1, 3, INPUT_HEIGHT, INPUT_WIDTH), (1, INPUT_HEIGHT, INPUT_WIDTH, 3),
                     (1, 768, INPUT_WIDTH, 1), (768, INPUT_WIDTH)):
        raise MetadataMismatchError("PP-LiteSeg requires 1024x512 NV12 geometry.")
    if meta.input_dtypes.get(input_name) != "nv12":
        raise MetadataMismatchError("PP-LiteSeg input must be NV12.")
    if meta.output_shapes.get(output_name) != (1, INPUT_HEIGHT, INPUT_WIDTH, 1):
        raise MetadataMismatchError("PP-LiteSeg expects a (1,512,1024,1) class map, not logits.")
    if meta.output_dtypes.get(output_name) != "int32":
        raise MetadataMismatchError("PP-LiteSeg class-map output must be int32.")
    return ModelBinding(selection, meta, input_name, output_name)


def create_runner(selection: ModelSelection, *, runtime_factory=None, runtime=None) -> SingleArrayRunner:
    """Construct the lazy PP-LiteSeg transport for a resolved selection.

    Args:
        selection: Manifest-backed selection carrying target, asset, and path.
        runtime_factory: Optional model-path-to-SDK-object factory (host seam).
        runtime: Optional prebuilt SDK object; overrides the factory.

    Returns:
        SingleArrayRunner: Lazy runner bound to the selected physical input;
        loading gates board identity and the published file hash.
    """
    return SingleArrayRunner(
        selection,
        binding_loader=bind_model,
        physical_input=lambda binding: ((1,3,INPUT_HEIGHT,INPUT_WIDTH), "float32") if binding.selection.float_logits else ((768, INPUT_WIDTH), "uint8"),
        task_name="PP-LiteSeg",
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
    """Prepared input tensors plus this call's geometry context.

    Attributes:
        tensors: Packed NV12 uint8 on X5; normalized NCHW float32 on S.
        context: Frozen source geometry of this call only.
    """

    tensors: Mapping[str, np.ndarray]
    context: ImageContext


class PPLiteSegSegmenter:
    """Segment one image into 19 Cityscapes classes with a compiled PP-LiteSeg.

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
        """Prepare packed NV12 on X5 or normalized float RGB on S.

        Args:
            image: uint8 BGR array shaped (H, W, 3), values [0, 255]; not
                modified. INTER_LINEAR stretches to 1024x512; no letterbox or
                CPU normalization on X5. S profiles use RGB/255, mean .5 and
                standard deviation .5 in contiguous float32 NCHW.

        Returns:
            PreparedInput: Input-name mapping of owned contiguous uint8 packed
            NV12 shaped (768, 1024), plus this call's frozen context.

        Raises:
            ValueError: The array shape, dtype, or dimensions are invalid.
        """
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or not all(image.shape[:2])
        ):
            raise ValueError('Expected a nonempty BGR HWC uint8 image.')
        resized = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        if self.binding.selection.float_logits:
            rgb=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB).astype(np.float32)
            tensor=np.ascontiguousarray(((rgb/255-.5)/.5).transpose(2,0,1)[None])
            return PreparedInput({self.binding.input_name:tensor},ImageContext(*image.shape[:2]))
        y, uv = bgr_to_nv12_planes(resized)
        packed = np.concatenate((y.reshape(-1), uv.reshape(-1))).reshape(768, INPUT_WIDTH)
        return PreparedInput(
            {self.binding.input_name: packed}, ImageContext(*image.shape[:2])
        )

    def infer(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Execute one inference call with the prepared physical tensors.

        Args:
            tensors: Input mapping returned by ``preprocess(...).tensors``.

        Returns:
            np.ndarray: Raw owned int32 class IDs shaped (1, 512, 1024, 1),
            unchanged: no argmax, softmax, dequantization, or decoding here.

        Raises:
            MetadataMismatchError: Input names, shapes, dtype, or the runtime
                output structure violate the binding.
            RuntimeError: SDK execution fails.
        """
        return self.runner(tensors)

    def postprocess(self, raw: np.ndarray) -> np.ndarray:
        """Return model-resolution labels using the selected output semantics.

        Args:
            raw: Output of ``infer``. X5 already returns an int32 class map.
                S returns float logits and applies one CPU argmax along the
                bound class axis, without softmax or dequantization.

        Returns:
            np.ndarray: Owned int32 ``(512, 1024)`` class IDs in [0, 18]. No
            context restoration occurs because the source contract returns
            model-resolution labels.

        Raises:
            ValueError: Logits, wrong dtype/shape, or out-of-range IDs.
        """
        if self.binding.selection.float_logits:
            expected=self.binding.metadata.output_shapes[self.binding.output_name]
            if not isinstance(raw,np.ndarray) or tuple(raw.shape)!=expected or raw.dtype!=np.float32 or not np.isfinite(raw).all():
                raise ValueError('Expected the bound finite float32 19-class logits tensor.')
            axis=1 if expected[1]==NUM_CLASSES else 3
            return np.argmax(raw,axis=axis)[0].astype(np.int32)
        if (
            not isinstance(raw, np.ndarray)
            or raw.shape != (1, INPUT_HEIGHT, INPUT_WIDTH, 1)
            or raw.dtype != np.int32
            or np.any(raw < 0)
            or np.any(raw > NUM_CLASSES - 1)
        ):
            raise ValueError(
                'Expected int32 class IDs 0..18 shaped (1,512,1024,1), not logits.'
            )
        return raw[0, :, :, 0].copy()

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run preprocessing, inference, and postprocessing for one image.

        Args:
            image: uint8 BGR array shaped (H, W, 3), values [0, 255].

        Returns:
            np.ndarray: int32 model-resolution class-ID map; see postprocess.

        Raises:
            ValueError: Image data, tensors, or class IDs are invalid.
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
