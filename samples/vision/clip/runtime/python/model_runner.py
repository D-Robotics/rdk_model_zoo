# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Lazy BPU image + CPU ONNX text adapter; returns unmodified raw features."""
from typing import Mapping
import numpy as np
from samples._shared.model_runner import _default_runtime_factory
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata
from samples.vision.clip.runtime.python.model_binding import bind_model


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
            from samples._shared.platforms import require_execution_target
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
