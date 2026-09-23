# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Lazy YOLOv5 SDK adapter; both fixed source platforms use named containers."""
from samples._shared.runtime_meta import RuntimeMetadata, MetadataMismatchError
from .model_binding import bind_model, validate_tensors


class RuntimeModelRunner:
    """Bind once; validate input/output without decoding or numerical conversion.

    A supplied runtime_factory is a host-fixture seam. Normal execution verifies
    hardware identity and artifact bytes before importing the SDK. Not thread safe.
    """
    def __init__(self,selection,*,runtime_factory=None):
        self.selection=selection;self._factory=runtime_factory;self.runtime=None;self.binding=None

    def load(self):
        """Return the exact binding; mismatch fails before any inference."""
        if self.binding is not None:return self.binding
        if self._factory is None:
            from samples._shared.platforms import require_execution_target
            from samples._shared.assets import verify_asset_file
            require_execution_target(self.selection.target)
            verify_asset_file(self.selection.asset,self.selection.model_path)
            from samples._shared.model_runner import _default_runtime_factory
            factory=_default_runtime_factory()
        else:factory=self._factory
        runtime=factory(str(self.selection.model_path))
        binding=bind_model(self.selection,RuntimeMetadata.from_runtime(runtime))
        self.runtime=runtime;self.binding=binding
        return binding

    def __call__(self,tensors):
        """Return flat native outputs; both sources call run({model_name: inputs})."""
        binding=self.load();inputs=validate_tensors(binding,tensors)
        result=self.runtime.run({binding.model_name:inputs})
        if not isinstance(result,dict) or set(result)!={binding.model_name}:raise MetadataMismatchError('Unexpected runtime model container.')
        return validate_tensors(binding,result[binding.model_name],outputs=True)

    def set_scheduling_params(self,*,priority=0,bpu_cores=None):
        """Pass native named-model priority/core values; no installation or fallback."""
        if type(priority) is not int or not 0<=priority<=255:raise ValueError('priority must be integer 0..255.')
        if bpu_cores is not None and (not isinstance(bpu_cores,(tuple,list)) or not bpu_cores or any(type(v) is not int or v<0 for v in bpu_cores)):raise ValueError('bpu_cores must be nonempty nonnegative integer indexes.')
        binding=self.load();kwargs={'priority':{binding.model_name:priority}}
        if bpu_cores is not None:kwargs['bpu_cores']={binding.model_name:list(bpu_cores)}
        self.runtime.set_scheduling_params(**kwargs)
