# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""CLIP image/text matching: load, preprocess, infer, postprocess, predict.

``CLIPMatcher`` owns the published X5 encoder-pair contract end to end:
construction loads the BPU image encoder and CPU ONNX text encoder through
the sample's lazy dual runtime, and each ``predict`` call runs
preprocess -> infer -> postprocess visible in this file. Catalog selection
and presentation live in ``cli.py``; the BPE tokenizer algorithm lives in
``tokenization.py``/``simple_tokenizer.py``; rendering lives in
``cli.py``.

Protocol provenance: ac115717197920355fc390bb04299b20e6436864,
CLIP conversion README and runtime. Tensor names are obtained from actual
metadata, since the source's protocol table gives conceptual names only.
"""
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from utils.py_utils.model_runner import _default_runtime_factory
from utils.py_utils.runtime_meta import MetadataMismatchError, RuntimeMetadata

from samples.vision.clip.runtime.python.cli import ModelSelection


@dataclass(frozen=True)
class ModelBinding:
    """Validated image/text encoder tensor protocol.

    Attributes:
        selection: The manifest-backed selection this binding was built from.
        image_model_name: BPU image-encoder submodel name.
        image_input_name: Bound F32 image input tensor name.
        image_output_name: Bound F32 image-feature output tensor name.
        text_input_name: Bound I32 token input tensor name.
        text_output_name: Bound F32 text-feature output tensor name.
        text_batch_size: Fixed ONNX batch when declared; None when dynamic.
    """

    selection: ModelSelection
    image_model_name: str
    image_input_name: str
    image_output_name: str
    text_input_name: str
    text_output_name: str
    text_batch_size: "int | None"


def _batch_dimension(dim):
    if dim is None or isinstance(dim, str):
        return None
    if type(dim) is int and dim > 0:
        return dim
    raise MetadataMismatchError(f'Invalid ONNX batch dimension {dim!r}.')


def bind_model(selection, image_metadata, text_session):
    """Validate F32 image features and I32-token/F32-feature ONNX tensors.

    Args:
        selection: Manifest-backed selection whose assets and paths must match.
        image_metadata: Board-observed image-encoder metadata.
        text_session: ONNX text-encoder session exposing input/output metadata.

    Returns:
        ModelBinding: Validated tensor names for both encoders.

    Raises:
        MetadataMismatchError: Selection or tensor metadata violates the
            contract.
    """
    from samples.vision.clip.runtime.python.cli import resolve_selection

    expected = resolve_selection(selection.target, image_asset_id=selection.image_asset.reference,
                                 text_asset_id=selection.text_asset.reference,
                                 image_model_path=selection.image_model_path,
                                 text_model_path=selection.text_model_path)
    if expected != selection:
        raise MetadataMismatchError('Encoder selection differs from publication facts.')
    facts = image_metadata if isinstance(image_metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(image_metadata)
    if len(facts.input_names) != 1 or len(facts.output_names) != 1:
        raise MetadataMismatchError('CLIP image encoder requires one input and one output.')
    inp, out = facts.input_names[0], facts.output_names[0]
    if facts.input_shapes.get(inp) != (1, 3, 224, 224) or facts.input_dtypes.get(inp) != 'float32':
        raise MetadataMismatchError('CLIP image input must be F32[1,3,224,224].')
    if facts.output_shapes.get(out) != (1, 512) or facts.output_dtypes.get(out) != 'float32':
        raise MetadataMismatchError('CLIP image output must be F32[1,512].')
    inputs, outputs = text_session.get_inputs(), text_session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise MetadataMismatchError('CLIP text encoder requires one input and one output.')
    text_in, text_out = inputs[0], outputs[0]
    for desc, dtype, width in ((text_in, 'tensor(int32)', 77), (text_out, 'tensor(float)', 512)):
        if not isinstance(desc.name, str) or not desc.name or desc.type != dtype or len(desc.shape) != 2 or desc.shape[1] != width:
            raise MetadataMismatchError(f'Invalid CLIP text tensor metadata: {desc.name!r}.')
    batches = [_batch_dimension(text_in.shape[0]), _batch_dimension(text_out.shape[0])]
    fixed = {size for size in batches if size is not None}
    if len(fixed) > 1:
        raise MetadataMismatchError('CLIP text input/output batch metadata conflicts.')
    return ModelBinding(selection, facts.model_name, inp, out, text_in.name,
                        text_out.name, next(iter(fixed), None))


def _default_text_factory(path):
    import onnxruntime
    return onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])


class RuntimeModelRunner:
    """Own both runtime lifecycles; injected runtimes are a host testing seam.

    Neither this runner nor the underlying SDK is declared thread-safe.
    Pre/postprocessing, tokenization and task result conversion live elsewhere.
    """

    def __init__(self, selection, *, image_runtime=None, text_session=None,
                 image_factory=None, text_factory=None):
        self.selection = selection
        self.image_runtime, self.text_session = image_runtime, text_session
        self.image_factory, self.text_factory = image_factory, text_factory
        self.binding = None

    @property
    def loaded(self):
        return self.binding is not None and self.image_runtime is not None and self.text_session is not None

    def load(self):
        """Gate the actual board before default loading and validate both models."""
        if self.loaded:
            return self.binding
        if ((self.image_runtime is None and self.image_factory is None) or
                (self.text_session is None and self.text_factory is None)):
            from utils.py_utils.platforms import require_execution_target
            require_execution_target(self.selection.target)
        try:
            if self.image_runtime is None:
                factory = self.image_factory or _default_runtime_factory()
                self.image_runtime = factory(str(self.selection.image_model_path))
            if self.text_session is None:
                factory = self.text_factory or _default_text_factory
                self.text_session = factory(str(self.selection.text_model_path))
            facts = RuntimeMetadata.from_runtime(self.image_runtime)
            self.binding = bind_model(self.selection, facts, self.text_session)
        except Exception:
            self.binding = self.image_runtime = self.text_session = None
            raise
        return self.binding

    def set_scheduling_params(self, *, priority=None, bpu_cores=None):
        """Schedule only the BPU image encoder, as in the fixed source."""
        if priority is not None and not 0 <= priority <= 255:
            raise ValueError('priority must be 0..255.')
        if bpu_cores is not None and (not bpu_cores or any(core < 0 for core in bpu_cores)):
            raise ValueError('bpu_cores must be a nonempty list of nonnegative indexes.')
        binding = self.load()
        kwargs = {}
        if priority is not None:
            kwargs['priority'] = {binding.image_model_name: priority}
        if bpu_cores is not None:
            kwargs['bpu_cores'] = {binding.image_model_name: list(bpu_cores)}
        if kwargs:
            self.image_runtime.set_scheduling_params(**kwargs)

    def __call__(self, tensors):
        """Validate both inputs before running either encoder; preserve arrays."""
        binding = self.load()
        if not isinstance(tensors, Mapping) or set(tensors) != {'image', 'texts'}:
            raise MetadataMismatchError('CLIP requires image and texts tensors.')
        image, texts = np.asarray(tensors['image']), np.asarray(tensors['texts'])
        if image.shape != (1, 3, 224, 224) or image.dtype != np.float32 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise MetadataMismatchError('CLIP image must be finite F32[1,3,224,224] in [0,1].')
        if texts.ndim != 2 or texts.shape[1] != 77 or texts.shape[0] < 1 or texts.dtype != np.int32 or texts.min() < 0 or texts.max() > 49407:
            raise MetadataMismatchError('CLIP tokens must be I32[N,77], N>0 and IDs 0..49407.')
        if binding.text_batch_size is not None and texts.shape[0] != binding.text_batch_size:
            raise MetadataMismatchError('Prompt count differs from fixed ONNX batch metadata.')
        try:
            output = self.image_runtime.run({binding.image_model_name: {binding.image_input_name: image}})
        except Exception as exc:
            raise RuntimeError(f'CLIP image encoder stage failed: {exc}') from exc
        if not isinstance(output, Mapping) or binding.image_model_name not in output:
            raise MetadataMismatchError('CLIP image runtime did not return the bound model.')
        flat = output[binding.image_model_name]
        if not isinstance(flat, Mapping) or set(flat) != {binding.image_output_name}:
            raise MetadataMismatchError('CLIP image runtime output names changed.')
        image_feature = np.asarray(flat[binding.image_output_name])
        if image_feature.shape != (1, 512) or image_feature.dtype != np.float32:
            raise MetadataMismatchError('CLIP image runtime output shape/dtype changed.')
        try:
            text_outputs = self.text_session.run([binding.text_output_name], {binding.text_input_name: texts})
        except Exception as exc:
            raise RuntimeError(f'CLIP text encoder stage failed: {exc}') from exc
        if not isinstance(text_outputs, (list, tuple)) or len(text_outputs) != 1:
            raise MetadataMismatchError('CLIP text runtime must return one output.')
        text_features = np.asarray(text_outputs[0])
        if text_features.shape != (texts.shape[0], 512) or text_features.dtype != np.float32:
            raise MetadataMismatchError('CLIP text runtime output shape/dtype changed.')
        return {'image_feature': image_feature, 'text_features': text_features}


@dataclass(frozen=True)
class InputContext:
    """Per-call image geometry and prompt identity.

    Attributes:
        original_shape: Source image (height, width).
        resized_shape: Aspect-preserving resized (height, width).
        crop_origin: Center-crop (y, x) origin of the 224 window.
        texts: Prompt strings of this call, in order.
    """

    original_shape: tuple
    resized_shape: tuple
    crop_origin: tuple
    texts: tuple


@dataclass(frozen=True)
class PreparedInput:
    """Owned multimodal tensors and immutable per-call context.

    Attributes:
        tensors: Semantic-key mapping with image F32[1,3,224,224] and
            texts I32[N,77]; the runner adapts keys to observed names.
        context: Frozen geometry and prompt identity of this call only.
    """

    tensors: Mapping[str, np.ndarray]
    context: InputContext


def prepare_inputs(image, texts, tokenizer):
    """Match source RGB/bicubic/rounded short-side/center crop/F32 divide.

    Args:
        image: uint8 BGR array shaped (H, W, 3); not modified.
        texts: Nonempty sequence of prompt strings.
        tokenizer: Callable producing I32[N,77] token arrays.

    Returns:
        PreparedInput: Image and token tensors plus this call's context.

    Raises:
        ValueError: The image or prompt list is invalid.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError('image must be a nonempty BGR uint8 H×W×3 ndarray.')
    tokens = tokenizer(texts)
    h, w = image.shape[:2]
    if h < w:
        new_h, new_w = 224, int(round(w * 224 / h))
    else:
        new_h, new_w = int(round(h * 224 / w)), 224
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    y, x = max((new_h - 224)//2, 0), max((new_w - 224)//2, 0)
    cropped = resized[y:y+224, x:x+224].astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(cropped.transpose(2, 0, 1)[None])
    return PreparedInput({'image': tensor, 'texts': tokens},
                         InputContext((h, w), (new_h, new_w), (y, x), tuple(texts)))


@dataclass(frozen=True)
class MatchResult:
    """Owned cosine scores and descending rank order.

    Attributes:
        scores: float32 array shaped (N,) of cosine similarities.
        order: Integer rank IDs with the best match first.
    """

    scores: np.ndarray
    order: np.ndarray


class CLIPMatcher:
    """Match one image against prompts with a compiled CLIP encoder pair.

    The constructor loads both encoders; injected runtimes/tokenizer are the
    documented host seams. No file/SDK/visualization I/O happens here.

    Attributes:
        runner (RuntimeModelRunner): Lazy dual-runtime runner used by infer.
        binding (ModelBinding): Validated encoder tensor protocol.
        tokenizer: Callable producing I32[N,77] token arrays.
    """

    def __init__(self, selection: ModelSelection, *, tokenizer=None, runner=None):
        """Load both encoders and validate their tensor protocols.

        Args:
            selection: Manifest-backed selection from ``cli.resolve_selection``.
            tokenizer: Optional injected tokenizer; defaults to the preserved
                BPE ``PromptTokenizer``.
            runner: Optional injected runner (host-test seam); defaults to
                the lazy dual runtime with the board-identity gate.

        Returns:
            None.

        Raises:
            TypeError: The injected runner or tokenizer is not callable.
            ValueError: The selection or contract is invalid.
            RuntimeError: Board identity or encoder loading fails.
        """
        if tokenizer is None:
            from samples.vision.clip.runtime.python.tokenization import PromptTokenizer
            tokenizer = PromptTokenizer()
        if not callable(tokenizer):
            raise TypeError('tokenizer must be callable.')
        self.tokenizer = tokenizer
        self.runner = runner if runner is not None else RuntimeModelRunner(selection)
        if not callable(self.runner):
            raise TypeError('runner must be callable.')
        self.binding = self.runner.load()

    def preprocess(self, image, texts):
        """Prepare source image bytes and BPE tokens, retaining request context.

        Args:
            image: uint8 BGR array shaped (H, W, 3); RGB converted, bicubic
                short-side resized to 224 (rounded), center cropped, and
                divided by 255.
            texts: Nonempty sequence of prompt strings.

        Returns:
            PreparedInput: Image F32 and token I32 tensors plus this call's
            frozen geometry and prompt identity.

        Raises:
            ValueError: The image or prompt list is invalid.
        """
        return prepare_inputs(image, texts, self.tokenizer)

    def infer(self, tensors):
        """Invoke the paired runner only; preserve the raw encoder outputs.

        Args:
            tensors: Mapping returned by ``preprocess(...).tensors``.

        Returns:
            Mapping with finite float32 ``image_feature`` (1, 512) and
            ``text_features`` (N, 512); no cosine math or ranking here.

        Raises:
            MetadataMismatchError: Input or output violates the binding.
            RuntimeError: Either encoder stage fails.
        """
        return self.runner(tensors)

    def postprocess(self, outputs):
        """Compute source F32 cosine with norm+1e-12 and descending argsort.

        Args:
            outputs: Raw output mapping returned by infer.

        Returns:
            MatchResult: Owned float32 scores (N,) and rank IDs; ties keep
            argsort's stable order reversed.

        Raises:
            ValueError: Container, shape, dtype, or finiteness is invalid.
        """
        if not isinstance(outputs, Mapping) or set(outputs) != {'image_feature', 'text_features'}:
            raise ValueError('CLIP requires both raw feature outputs.')
        image = np.asarray(outputs['image_feature'])
        texts = np.asarray(outputs['text_features'])
        if image.shape != (1, 512) or texts.ndim != 2 or texts.shape[1] != 512 or texts.shape[0] < 1:
            raise ValueError('CLIP feature output shapes are invalid.')
        if image.dtype != np.float32 or texts.dtype != np.float32 or not np.isfinite(image).all() or not np.isfinite(texts).all():
            raise ValueError('CLIP feature outputs must be finite float32.')
        image = image.reshape(-1).astype(np.float32)
        texts = texts.astype(np.float32)
        image_norm = np.linalg.norm(image) + 1e-12
        text_norm = np.linalg.norm(texts, axis=1) + 1e-12
        scores = (texts @ image) / (text_norm * image_norm)
        if not np.isfinite(scores).all():
            raise ValueError('CLIP cosine overflowed; check runtime feature values.')
        return MatchResult(scores, np.argsort(scores)[::-1])

    def predict(self, image, texts):
        """Compose exactly preparation → raw runner → cosine/ranking.

        Args:
            image: uint8 BGR array shaped (H, W, 3).
            texts: Nonempty sequence of prompt strings.

        Returns:
            MatchResult: Cosine scores and ranking; see postprocess.

        Raises:
            ValueError: Input data, tensors, or features are invalid.
            MetadataMismatchError: Tensor structure violates the binding.
            RuntimeError: Either encoder stage fails.
        """
        prepared = self.preprocess(image, texts)
        return self.postprocess(self.infer(prepared.tensors))

    def set_scheduling_params(self, *, priority=None, bpu_cores=None) -> None:
        """Schedule only the BPU image encoder, as in the fixed source.

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

    def pre_process(self, image, texts):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image, texts)

    def forward(self, tensors):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(tensors)

    def post_process(self, outputs):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs)

    def __call__(self, image, texts):
        """Delegate to predict with the same input and error contract."""
        return self.predict(image, texts)


__all__ = ['CLIPMatcher', 'MatchResult', 'PreparedInput',
           'RuntimeModelRunner', 'bind_model', 'prepare_inputs']
