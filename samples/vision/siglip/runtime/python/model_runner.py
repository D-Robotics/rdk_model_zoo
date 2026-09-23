# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Lazy packed SigLIP SDK adapter; metadata/containers only, no task transforms."""
from dataclasses import replace
from typing import Mapping
import numpy as np
from samples._shared.runtime_meta import RuntimeMetadata, MetadataMismatchError
from samples._shared.model_runner import _default_runtime_factory
from samples.vision.siglip.runtime.python.model_binding import SUBMODELS, bind_model


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
            from samples._shared.platforms import require_execution_target
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
        if priority is not None:kwargs['priority'] = dict.fromkeys(SUBMODELS, priority)
        if bpu_cores is not None:kwargs['bpu_cores'] = {s:list(bpu_cores) for s in SUBMODELS}
        if kwargs:self._runtime.set_scheduling_params(**kwargs)

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
