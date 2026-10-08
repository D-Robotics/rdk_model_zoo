# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv2 feature embedding: load, preprocess, infer, postprocess, predict.

``DINOv2Embedder`` owns the fixed dual-output contract end to end:
construction loads the model through the sample's lazy runner, and each
``predict`` call runs preprocess -> infer -> postprocess visible in this
file. Catalog selection and reporting live in ``cli.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from utils.py_utils.model_runner import _default_runtime_factory
from utils.py_utils.quantization import apply_output_transform, validate_output_transform
from utils.py_utils.runtime_meta import (
    MetadataMismatchError,
    RuntimeMetadata,
    canonicalise_dtype,
)

from samples.vision.dinov2.runtime.python.cli import OUTPUTS, OUTPUT_SHAPES, ModelSelection

CLS_FEAT = "cls_feat"
PATCH_FEAT = "patch_feat"
NUMERIC_DTYPES = {"int8", "int16", "int32", "uint8", "uint16", "float32"}

IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class ModelBinding:
    """Validated DINOv2 tensor protocol for the dual-output artifact.

    Attributes:
        selection: The manifest-backed selection this binding was built from.
        model_name: Single submodel name used for SDK run calls.
        input_name: Fixed input tensor name ``input``.
        input_shape: Fixed input shape ``(1, 3, 224, 224)``.
        output_names: Bound output names ``(cls_feat, patch_feat)``.
        output_shapes: Exact observed output shapes keyed by name.
        output_dtypes: Canonicalised output dtypes keyed by name.
        output_quants: SDK quantization descriptors keyed by name.
        output_transforms: Per-output ``raw_f32``/``dequant`` decisions.
    """

    selection: ModelSelection
    model_name: str
    input_name: str
    input_shape: tuple[int, ...]
    output_names: tuple[str, ...]
    output_shapes: dict[str, tuple[int, ...]]
    output_dtypes: dict[str, str]
    output_quants: dict[str, Any]
    output_transforms: dict[str, str]


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    """Validate actual runtime metadata against the two-output DINO contract.

    Args:
        selection: Manifest-backed selection whose asset and path must match.
        metadata: Board-observed metadata mapping or ``RuntimeMetadata``.

    Returns:
        ModelBinding: Validated names, shapes, and per-output transforms.

    Raises:
        ValueError: The selection does not match the manifest publication.
        MetadataMismatchError: Tensor names, shapes, or dtypes violate the
            contract.
    """
    from samples.vision.dinov2.runtime.python.cli import resolve_selection

    expected = resolve_selection(
        selection.target,
        asset_id=selection.asset.reference,
        model_path=selection.model_path,
    )
    if expected.asset != selection.asset:
        raise ValueError("Selection publication facts do not match the manifest.")
    facts = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if facts.input_names != ("input",) or facts.input_shapes.get("input") != (1, 3, 224, 224):
        raise MetadataMismatchError("DINOv2 input must be input F32 (1,3,224,224).")
    if facts.input_dtypes.get("input") != "float32":
        raise MetadataMismatchError("DINOv2 input dtype must be float32.")
    if facts.output_names != OUTPUTS:
        raise MetadataMismatchError(f"DINOv2 outputs must be {OUTPUTS}; got {facts.output_names}.")

    dtypes: dict[str, str] = {}
    transforms: dict[str, str] = {}
    for name in OUTPUTS:
        shape = facts.output_shapes.get(name)
        if shape != OUTPUT_SHAPES[name]:
            raise MetadataMismatchError(f"DINOv2 {name} shape must be {OUTPUT_SHAPES[name]}; got {shape}.")
        dtype = canonicalise_dtype(facts.output_dtypes.get(name))
        if dtype not in NUMERIC_DTYPES:
            raise MetadataMismatchError(f"DINOv2 {name} dtype is unsupported: {dtype!r}.")
        dtypes[name] = dtype
        if dtype == "float32":
            transforms[name] = validate_output_transform("raw_f32")
        else:
            if name not in facts.output_quants:
                raise MetadataMismatchError(f"DINOv2 {name} integer output has no quantization descriptor.")
            transforms[name] = validate_output_transform("dequant")
    return ModelBinding(
        selection=selection,
        model_name=facts.model_name,
        input_name="input",
        input_shape=(1, 3, 224, 224),
        output_names=OUTPUTS,
        output_shapes=dict(facts.output_shapes),
        output_dtypes=dtypes,
        output_quants=dict(facts.output_quants),
        output_transforms=transforms,
    )


class RuntimeModelRunner:
    """Load and validate one selected HBM lazily; instances are not thread-safe.

    The board identity gate runs before the SDK import on the real path; an
    injected ``runtime``/``runtime_factory`` is the documented host seam.
    """

    def __init__(self, selection: ModelSelection, *, runtime: Any = None, runtime_factory: Any = None):
        self.selection = selection
        self._runtime = runtime
        self._factory = runtime_factory
        self.binding: ModelBinding | None = None
        self.metadata: RuntimeMetadata | None = None

    @property
    def loaded(self) -> bool:
        return self._runtime is not None and self.binding is not None

    @property
    def runtime(self) -> Any:
        if self._runtime is None:
            raise RuntimeError("Runtime has not been loaded; call load() first.")
        return self._runtime

    def load(self) -> ModelBinding:
        """Gate hardware, construct SDK runtime, and bind actual metadata."""

        if self.loaded:
            return self.binding  # type: ignore[return-value]
        if self._runtime is None and self._factory is None:
            from utils.py_utils.platforms import require_execution_target

            require_execution_target(self.selection.target)
        try:
            if self._runtime is None:
                factory = self._factory or _default_runtime_factory()
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
        """Apply source scheduling options after metadata binding."""

        if priority is not None and not 0 <= priority <= 255:
            raise ValueError("priority must be between 0 and 255")
        if bpu_cores is not None and (not bpu_cores or any(core < 0 for core in bpu_cores)):
            raise ValueError("bpu_cores must be a nonempty list of nonnegative indexes")
        binding = self.load()
        if priority is None and bpu_cores is None:
            return
        setter = getattr(self.runtime, "set_scheduling_params", None)
        if not callable(setter):
            raise RuntimeError("The installed runtime does not expose scheduling parameters")
        kwargs: dict[str, Any] = {}
        if priority is not None:
            kwargs["priority"] = {binding.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {binding.model_name: list(bpu_cores)}
        setter(**kwargs)

    def __call__(self, tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Run one validated tensor mapping and return raw outputs unchanged."""

        binding = self.load()
        if not isinstance(tensors, Mapping) or set(tensors) != {binding.input_name}:
            raise MetadataMismatchError("DINOv2 requires exactly the input tensor named 'input'.")
        value = np.asarray(tensors[binding.input_name])
        if value.shape != binding.input_shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise MetadataMismatchError("DINOv2 input must be finite contiguous-shape float32 NCHW.")
        outputs = self.runtime.run({binding.model_name: dict(tensors)})
        if not isinstance(outputs, Mapping) or binding.model_name not in outputs:
            raise MetadataMismatchError(f"Runtime output is missing model {binding.model_name!r}.")
        flat = outputs[binding.model_name]
        if not isinstance(flat, Mapping) or set(flat) != set(binding.output_names):
            raise MetadataMismatchError("Runtime output does not contain exactly cls_feat and patch_feat.")
        result: dict[str, np.ndarray] = {}
        for name in binding.output_names:
            output = np.asarray(flat[name])
            if output.shape != binding.output_shapes[name] or output.dtype != np.dtype(binding.output_dtypes[name]):
                raise MetadataMismatchError(f"Runtime output {name!r} differs from bound shape/dtype.")
            result[name] = output
        return result


@dataclass(frozen=True)
class ImageContext:
    """Realized geometry for one preprocessing call.

    Attributes:
        original_shape: Source image (height, width).
        resized_shape: Aspect-preserving resized (height, width).
        crop_origin: Center-crop (y, x) origin of the 224 window.
    """

    original_shape: tuple[int, int]
    resized_shape: tuple[int, int]
    crop_origin: tuple[int, int]


@dataclass(frozen=True)
class PreparedInput:
    """Owned physical tensors and immutable per-call context.

    Attributes:
        tensors: Input-name to normalized F32 NCHW tensor mapping.
        context: Frozen realized geometry of this call only.
    """

    tensors: Mapping[str, np.ndarray]
    context: ImageContext


def prepare_image(image: np.ndarray, binding: ModelBinding) -> PreparedInput:
    """Convert BGR uint8 HWC to the fixed DINOv2 float32 NCHW input.

    Args:
        image: uint8 BGR array shaped (H, W, 3); not modified.
        binding: Validated DINOv2 binding supplying the input contract.

    Returns:
        PreparedInput: Input-name mapping of one contiguous float32
        (1, 3, 224, 224) tensor normalized with ImageNet statistics, plus
        this call's frozen geometry.

    Raises:
        ValueError: The image or binding does not describe the fixed
            contract.
    """
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 1
    ):
        raise ValueError("Expected nonempty BGR uint8 image shaped HxWx3.")
    if binding.input_name != "input" or binding.input_shape != (1, 3, 224, 224):
        raise ValueError("DINOv2 binding does not describe the fixed input contract.")

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if height <= width:
        resized_height = RESIZE_SIZE
        resized_width = int(RESIZE_SIZE * width / height)
    else:
        resized_height = int(RESIZE_SIZE * height / width)
        resized_width = RESIZE_SIZE
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_CUBIC,
    )
    crop_y = (resized_height - IMAGE_SIZE) // 2
    crop_x = (resized_width - IMAGE_SIZE) // 2
    cropped = resized[crop_y:crop_y + IMAGE_SIZE, crop_x:crop_x + IMAGE_SIZE]
    tensor = np.transpose(cropped, (2, 0, 1))[None].astype(np.float32)
    tensor = tensor / 255.0
    tensor = (tensor - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
    tensor = np.ascontiguousarray(tensor)
    if tensor.shape != binding.input_shape or tensor.dtype != np.float32:
        raise ValueError(f"Preprocessed input is {tensor.shape}/{tensor.dtype}, expected fixed F32 input.")
    return PreparedInput(
        tensors={binding.input_name: tensor},
        context=ImageContext((height, width), (resized_height, resized_width), (crop_y, crop_x)),
    )


class DINOv2Embedder:
    """Embed one image with a compiled DINOv2 model.

    The constructor accepts a resolved selection; see __init__ for the
    injection seam. Construction loads the model immediately.

    Attributes:
        runner (RuntimeModelRunner): Lazy sample runner used by infer.
        binding (ModelBinding): Validated tensor protocol and transforms.
        output: Selected output name, ``cls_feat`` or ``patch_feat``.
    """

    def __init__(self, selection: ModelSelection, *, output: str = CLS_FEAT, runner=None) -> None:
        """Load the compiled model and validate its tensor protocol.

        Args:
            selection: Manifest-backed selection from ``cli.resolve_selection``.
            output: Output to decode: ``cls_feat`` (default) or ``patch_feat``.
            runner: Optional injected runner (host-test seam); defaults to
                the lazy sample runner with the board-identity gate.

        Returns:
            None.

        Raises:
            TypeError: The injected runner is not callable.
            ValueError: The output name or contract is unsupported.
            RuntimeError: Board identity or SDK loading fails.
        """
        if output not in OUTPUTS:
            raise ValueError(f"Unsupported DINOv2 output: {output}")
        self.runner = runner if runner is not None else RuntimeModelRunner(selection)
        if not callable(self.runner):
            raise TypeError("runner must be callable")
        self.binding = self.runner.load()
        self.output = output

    def preprocess(self, image: np.ndarray, image_format: str = "BGR") -> PreparedInput:
        """Validate a BGR uint8 image and build one per-call prepared input.

        Args:
            image: uint8 BGR array shaped (H, W, 3); resized to 256 on the
                short side with INTER_CUBIC, center-cropped to 224, RGB
                converted, and normalized with ImageNet statistics.
            image_format: Only ``BGR`` is accepted.

        Returns:
            PreparedInput: Normalized F32 NCHW tensors plus this call's
            frozen geometry.

        Raises:
            ValueError: The format or image data is invalid.
        """
        if image_format != "BGR":
            raise ValueError(f"Unsupported image_format: {image_format}")
        return prepare_image(image, self.binding)

    def infer(self, tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Call the runner and preserve its raw output values and containers.

        Args:
            tensors: Input mapping returned by ``preprocess(...).tensors``.

        Returns:
            Mapping[str, np.ndarray]: Raw ``cls_feat``/``patch_feat`` arrays
            matching the bound shapes and dtypes; no dequantization here.

        Raises:
            MetadataMismatchError: Input or output structure violates the
                binding.
            RuntimeError: SDK execution fails.
        """
        return self.runner(tensors)

    def postprocess(self, outputs: Mapping[str, np.ndarray]) -> np.ndarray:
        """Dequantize the selected output and return an owned F32 array.

        Args:
            outputs: Raw output mapping returned by infer, containing exactly
                both bound outputs.

        Returns:
            np.ndarray: Owned float32 feature tensor for the selected
            output; (1, 384) for cls_feat or (1, 256, 384) for patch_feat.
            No activation is applied.

        Raises:
            ValueError: Container, shape, dtype, or finiteness is invalid.
        """
        if not isinstance(outputs, Mapping) or set(outputs) != set(OUTPUTS):
            raise ValueError("DINOv2 raw output must contain exactly cls_feat and patch_feat.")
        raw_by_name: dict[str, np.ndarray] = {}
        for name in OUTPUTS:
            value = np.asarray(outputs[name])
            if value.shape != self.binding.output_shapes[name]:
                raise ValueError(f"DINOv2 {name} shape differs from bound metadata.")
            if value.dtype != np.dtype(self.binding.output_dtypes[name]):
                raise ValueError(f"DINOv2 {name} dtype differs from bound metadata.")
            raw_by_name[name] = value
        name = self.output
        transformed = apply_output_transform(
            self.binding.output_transforms[name],
            {name: raw_by_name[name]},
            {name: self.binding.output_quants.get(name)},
        )[name]
        result = np.asarray(transformed).astype(np.float32, copy=True)
        if not np.isfinite(result).all():
            raise ValueError(f"DINOv2 {name} output contains NaN or Inf values.")
        return result

    def predict(self, image: np.ndarray, image_format: str = "BGR") -> np.ndarray:
        """Compose exactly preprocess -> infer -> postprocess.

        Args:
            image: uint8 BGR array shaped (H, W, 3).
            image_format: Only ``BGR`` is accepted.

        Returns:
            np.ndarray: Owned float32 feature tensor; see postprocess.

        Raises:
            ValueError: Image data, tensors, or features are invalid.
            MetadataMismatchError: Tensor structure violates the binding.
            RuntimeError: SDK execution fails.
        """
        prepared = self.preprocess(image, image_format)
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

    def pre_process(self, image: np.ndarray, image_format: str = "BGR") -> PreparedInput:
        """Delegate to preprocess with the same input and error contract."""
        return self.preprocess(image, image_format)

    def forward(self, tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Delegate to infer with the same input and error contract."""
        return self.infer(tensors)

    def post_process(self, outputs: Mapping[str, np.ndarray]) -> np.ndarray:
        """Delegate to postprocess with the same input and error contract."""
        return self.postprocess(outputs)

    def __call__(self, image: np.ndarray, image_format: str = "BGR") -> np.ndarray:
        """Delegate to predict with the same input and error contract."""
        return self.predict(image, image_format)


__all__ = ["CLS_FEAT", "DINOv2Embedder", "PATCH_FEAT",
           "PreparedInput", "RuntimeModelRunner", "bind_model", "prepare_image"]
