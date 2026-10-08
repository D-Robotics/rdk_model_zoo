# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""SigLIP feature embedding: load, preprocess, infer, postprocess, predict.

``SigLIPEmbedder`` owns the packed dual-submodel contract end to end:
construction loads both submodel bindings through the sample's lazy runner,
and each ``predict`` call runs preprocess -> infer -> postprocess visible in
this file. Catalog selection and reporting live in ``cli.py``.
"""

from dataclasses import dataclass, replace
from typing import Any, Mapping
import numpy as np

from utils.py_utils.runtime_meta import RuntimeMetadata, MetadataMismatchError
from utils.py_utils.model_runner import _default_runtime_factory

from samples.vision.siglip.runtime.python.cli import SUBMODELS, VARIANTS, ModelSelection


class ModelBinding:
    """Validated physical shapes/dtype for one selected submodel; no decoding.

    Attributes:
        selection: The manifest-backed selection this binding was built from.
        model_name: Selected packed submodel name.
        input_shape: Fixed F32 input shape ``(1, 3, size, size)``.
        output_shape: Observed output shape; no dequantization or squeeze.
        output_dtype: Observed native numeric output dtype.
    """

    def __init__(self, selection: ModelSelection, model_name: str,
                 input_shape: tuple[int, ...], output_shape: tuple[int, ...],
                 output_dtype: str):
        self.selection = selection
        self.model_name = model_name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_dtype = output_dtype


NUMERIC_DTYPES = ('float16', 'float32', 'float64', 'int8', 'uint8', 'int16', 'int32', 'int64')


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    """Validate both-submodel presence and selected input/output metadata.

    Args:
        selection: Manifest-backed selection whose asset and path must match.
        metadata: Board-observed metadata mapping or ``RuntimeMetadata``.

    Returns:
        ModelBinding: Validated input/output shapes and native dtype.

    Raises:
        ValueError: The selection does not match the manifest publication.
        MetadataMismatchError: Submodel, tensor, shape, or dtype contract
            fails.
    """
    from samples.vision.siglip.runtime.python.cli import resolve_selection

    expected = resolve_selection(selection.target, variant=selection.variant,
        asset_id=selection.asset.reference, model_path=selection.model_path,
        submodel=selection.submodel, image_size=selection.image_size)
    if expected.asset != selection.asset:
        raise ValueError('Selection publication facts do not match manifest.')
    m = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if set(m.model_names) != set(SUBMODELS) or m.model_name != selection.submodel:
        raise MetadataMismatchError('Expected both SigLIP packed submodels and the explicit selected model.')
    size, dim, count = VARIANTS[selection.variant]
    input_shape = (1, 3, size, size)
    if m.input_names != ('_input_0',) or m.input_shapes.get('_input_0') != input_shape or m.input_dtypes.get('_input_0') != 'float32':
        raise MetadataMismatchError(f'SigLIP input must be _input_0 F32 {input_shape}.')
    allowed = ((1, dim), (1, 1, dim)) if selection.submodel == 'pooler_output' else ((1, count, dim),)
    shape = m.output_shapes.get('_output_0')
    dtype = m.output_dtypes.get('_output_0')
    if m.output_names != ('_output_0',) or shape not in allowed or dtype not in NUMERIC_DTYPES:
        raise MetadataMismatchError(f'SigLIP output must be _output_0 numeric {allowed}; got {shape}/{dtype}.')
    return ModelBinding(selection, m.model_name, input_shape, shape, dtype)


class RuntimeModelRunner:
    """Load both submodel contracts once; execute the explicitly selected one.

    Explicit runtime/factory injection supports host tests only and does not
    certify board execution. Instances and their SDK are not declared thread-safe.
    """

    def __init__(self, selection, *, runtime=None, runtime_factory=None):
        self.selection = selection
        self._runtime = runtime
        self._factory = runtime_factory
        self.binding = None

    @property
    def loaded(self):
        return self._runtime is not None and self.binding is not None

    def load(self):
        """Gate actual hardware before default SDK loading; bind both submodels."""
        if self.loaded:
            return self.binding
        if self._runtime is None and self._factory is None:
            from utils.py_utils.platforms import require_execution_target
            require_execution_target(self.selection.target)
        try:
            if self._runtime is None:
                self._runtime = (self._factory or _default_runtime_factory())(str(self.selection.model_path))
            bindings = {}
            for name in SUBMODELS:
                metadata = RuntimeMetadata.from_runtime(self._runtime, model_name=name)
                bindings[name] = bind_model(replace(self.selection, submodel=name), metadata)
            self.binding = bindings[self.selection.submodel]
        except Exception:
            self._runtime = None
            self.binding = None
            raise
        return self.binding

    def set_scheduling_params(self, *, priority=None, bpu_cores=None):
        """Apply source scheduling to both packed submodels, before inference."""
        if priority is not None and not 0 <= priority <= 255:
            raise ValueError('priority must be 0..255')
        if bpu_cores is not None and (not bpu_cores or any(c < 0 for c in bpu_cores)):
            raise ValueError('bpu_cores must be a nonempty list of nonnegative indexes')
        self.load()
        kwargs = {}
        if priority is not None:
            kwargs['priority'] = dict.fromkeys(SUBMODELS, priority)
        if bpu_cores is not None:
            kwargs['bpu_cores'] = {s: list(bpu_cores) for s in SUBMODELS}
        if kwargs:
            self._runtime.set_scheduling_params(**kwargs)

    def __call__(self, inputs):
        """Validate F32 input and native output; return unchanged raw tensor."""
        binding = self.load()
        if not isinstance(inputs, Mapping) or set(inputs) != {'_input_0'}:
            raise MetadataMismatchError('SigLIP requires only _input_0.')
        value = np.asarray(inputs['_input_0'])
        if value.shape != binding.input_shape or value.dtype != np.float32 or not np.isfinite(value).all() or value.min() < -1 or value.max() > 1:
            raise MetadataMismatchError('SigLIP input must match bound F32 shape and [-1,1] range.')
        outputs = self._runtime.run({binding.model_name: dict(inputs)})
        if not isinstance(outputs, Mapping) or binding.model_name not in outputs:
            raise MetadataMismatchError('Missing selected SigLIP model in runtime output.')
        flat = outputs[binding.model_name]
        if not isinstance(flat, Mapping) or set(flat) != {'_output_0'}:
            raise MetadataMismatchError('Missing SigLIP _output_0 tensor.')
        output = np.asarray(flat['_output_0'])
        if output.shape != binding.output_shape or output.dtype != np.dtype(binding.output_dtype):
            raise MetadataMismatchError('SigLIP runtime output differs from bound shape/dtype.')
        return {'_output_0': output}


@dataclass(frozen=True)
class ImageContext:
    """Per-call geometry: source shape, resized shape, and letterbox padding.

    Attributes:
        original_shape: Source image (height, width).
        resized_shape: Aspect-preserving resized (height, width).
        padding: (top, bottom, left, right) zero-contribution padding.
    """

    original_shape: tuple[int, int]
    resized_shape: tuple[int, int]
    padding: tuple[int, int, int, int]


@dataclass(frozen=True)
class PreparedInput:
    """Owned F32 NCHW input and immutable geometry for one image.

    Attributes:
        tensors: ``_input_0`` to contiguous float32 NCHW tensor mapping.
        context: Frozen realized geometry of this call only.
    """

    tensors: Mapping[str, np.ndarray]
    context: ImageContext


def prepare_image(image: np.ndarray, size: int) -> PreparedInput:
    """BGR U8 H×W×3 to RGB F32 [1,3,size,size], range [-1,1].

    Args:
        image: uint8 BGR array shaped (H, W, 3); not modified.
        size: Square model input size from the selected variant.

    Returns:
        PreparedInput: ``_input_0`` tensor and this call's frozen geometry.
        Source numerical order is preserved: AREA resize, integer floor with
        minimum one pixel, padding 127, transpose/cast, then /127.5 - 1.

    Raises:
        ValueError: For non-U8, empty or non-three-channel input.
    """
    import cv2

    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError('Expected nonempty BGR uint8 image shaped H×W×3.')
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(int(h * scale), 1), max(int(w * scale), 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    dh, dw = size - nh, size - nw
    padding = (dh // 2, dh - dh // 2, dw // 2, dw - dw // 2)
    padded = cv2.copyMakeBorder(resized, *padding, cv2.BORDER_CONSTANT, value=(127, 127, 127))
    tensor = np.transpose(padded, (2, 0, 1))[None].astype(np.float32)
    tensor = tensor / 127.5 - 1.0
    return PreparedInput({'_input_0': np.ascontiguousarray(tensor)}, ImageContext((h, w), (nh, nw), padding))


class SigLIPEmbedder:
    """Embed one image with a compiled SigLIP packed-submodel artifact.

    The constructor accepts a resolved selection; see __init__ for the
    injection seam. Construction loads both submodel bindings immediately.

    Attributes:
        runner (RuntimeModelRunner): Lazy dual-submodel runner used by infer.
        binding (ModelBinding): Validated selected-submodel tensor protocol.
    """

    def __init__(self, selection: ModelSelection, *, runner=None) -> None:
        """Load the compiled model and validate both packed submodels.

        Args:
            selection: Manifest-backed selection from ``cli.resolve_selection``.
            runner: Optional injected runner (host-test seam); defaults to
                the lazy dual-submodel runner with the board-identity gate.

        Returns:
            None.

        Raises:
            TypeError: The injected runner is not callable.
            ValueError: The selection or contract is invalid.
            RuntimeError: Board identity or SDK loading fails.
        """
        self.runner = runner if runner is not None else RuntimeModelRunner(selection)
        if not callable(self.runner):
            raise TypeError('runner must be callable')
        self.binding = self.runner.load()

    def preprocess(self, image: np.ndarray) -> PreparedInput:
        """Validate a BGR image and build F32 tensors with per-call context.

        Args:
            image: uint8 BGR array shaped (H, W, 3); RGB converted, AREA
                resized with 127 letterbox padding, and scaled to [-1, 1].

        Returns:
            PreparedInput: RGB F32 NCHW tensors plus this call's frozen
            geometry.

        Raises:
            ValueError: The image data is invalid.
        """
        return prepare_image(image, self.binding.selection.image_size)

    def infer(self, tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Invoke the runner only, returning its raw mapping without numeric changes.

        Args:
            tensors: Input mapping returned by ``preprocess(...).tensors``.

        Returns:
            Mapping[str, np.ndarray]: Raw ``_output_0`` tensor in the bound
            native shape and dtype; no dequantization, activation, or
            squeeze here.

        Raises:
            MetadataMismatchError: Input or output violates the binding.
            RuntimeError: SDK execution fails.
        """
        return self.runner(tensors)

    def postprocess(self, outputs: Mapping[str, np.ndarray]) -> np.ndarray:
        """Validate shape/dtype/finiteness and own the selected raw embedding.

        Args:
            outputs: Raw output mapping returned by infer, containing only
                ``_output_0``.

        Returns:
            np.ndarray: Owned copy of the raw embedding in the bound native
            numeric dtype and shape; a copy prevents later SDK calls from
            mutating an already returned result.

        Raises:
            ValueError: Container, shape, dtype, or finiteness is invalid.
        """
        if not isinstance(outputs, Mapping) or set(outputs) != {'_output_0'}:
            raise ValueError('SigLIP output must contain only _output_0.')
        result = np.asarray(outputs['_output_0'])
        if result.shape != self.binding.output_shape or result.dtype != np.dtype(self.binding.output_dtype):
            raise ValueError('SigLIP output shape/dtype differs from bound metadata.')
        if not np.isfinite(result).all():
            raise ValueError('SigLIP output contains NaN or Inf.')
        return result.copy()

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Compose exactly preprocess -> infer -> postprocess.

        Args:
            image: uint8 BGR array shaped (H, W, 3).

        Returns:
            np.ndarray: Owned raw embedding; see postprocess.

        Raises:
            ValueError: Image data, tensors, or embedding values are invalid.
            MetadataMismatchError: Tensor structure violates the binding.
            RuntimeError: SDK execution fails.
        """
        prepared = self.preprocess(image)
        return self.postprocess(self.infer(prepared.tensors))

    def set_scheduling_params(self, *, priority=None, bpu_cores=None) -> None:
        """Apply source scheduling to both packed submodels, before inference.

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
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image)

    def forward(self, tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Compatibility alias for :meth:`infer`."""
        return self.infer(tensors)

    def post_process(self, outputs: Mapping[str, np.ndarray]) -> np.ndarray:
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Delegate to predict with the same input and error contract."""
        return self.predict(image)


__all__ = ['ModelBinding', 'PreparedInput', 'RuntimeModelRunner',
           'SigLIPEmbedder', 'bind_model', 'prepare_image']
