# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Lazy, stage-attributed SDK adapters shared by EfficientSAM and MobileSAM."""
from samples._shared.model_runner import _default_runtime_factory
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata
from samples._shared.sam_binding import bind_model, validate_tensors
from collections.abc import Mapping


class StageRunner:
    """Call one bound SDK model; preserve native outputs and numeric semantics."""
    def __init__(self, runtime, binding):
        self.runtime, self.binding = runtime, binding

    def __call__(self, tensors):
        binding = self.binding
        try:
            inputs = validate_tensors(binding, tensors)
            physical = inputs if binding.target == 'x5' else {binding.model_name: inputs}
            outputs = self.runtime.run(physical)
            if not isinstance(outputs, Mapping) or set(outputs) != {binding.model_name}:
                raise MetadataMismatchError(f'Expected model output container {binding.model_name!r}.')
            return validate_tensors(binding, outputs[binding.model_name], outputs=True)
        except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
            error = MetadataMismatchError if isinstance(exc, (ValueError, TypeError, KeyError)) else RuntimeError
            raise error(f'{binding.stage} stage failed: {exc}') from exc


class RuntimeModelRunner:
    """Load and bind a complete encoder/decoder pair before any inference.

    Factory injection is for host fixtures. Normal use gates actual hardware
    before importing the SDK. Instances are not advertised as thread-safe;
    stage post_process copies outputs before another SDK call can reuse them.
    """
    def __init__(self, selection, *, runtime_factory=None):
        self.selection = selection
        self._factory = runtime_factory
        self.binding = None
        self._encoder = None
        self._decoder = None

    @property
    def loaded(self):
        return self.binding is not None

    @property
    def encoder(self):
        if self._encoder is None:
            raise RuntimeError('Encoder is unavailable; load and bind both models first.')
        return self._encoder

    @property
    def decoder(self):
        if self._decoder is None:
            raise RuntimeError('Decoder is unavailable; load and bind both models first.')
        return self._decoder

    def load(self):
        """Return the pair binding, rejecting incomplete or mismatched metadata."""
        if self.loaded:
            return self.binding
        if self._factory is None:
            from samples._shared.platforms import require_execution_target
            require_execution_target(self.selection.target)
        stage = 'encoder'
        try:
            factory = self._factory or _default_runtime_factory()
            encoder = factory(str(self.selection.encoder_model_path))
            enc_meta = RuntimeMetadata.from_runtime(encoder)
            stage = 'decoder'
            decoder = factory(str(self.selection.decoder_model_path))
            dec_meta = RuntimeMetadata.from_runtime(decoder)
            # bind_model errors already identify the offending stage.
            stage = 'pair binding'
            binding = bind_model(self.selection, enc_meta, dec_meta)
            self._encoder = StageRunner(encoder, binding.encoder)
            self._decoder = StageRunner(decoder, binding.decoder)
            self.binding = binding
        except Exception as exc:
            self.binding = self._encoder = self._decoder = None
            error = MetadataMismatchError if isinstance(exc, (ValueError, KeyError, TypeError)) else RuntimeError
            raise error(f'{stage} load failed: {exc}') from exc
        return binding

    def set_scheduling_params(self, *, priority=None, bpu_cores=None):
        """Apply native scheduling to both stages; explicit X5 cores are invalid."""
        if priority is not None and (type(priority) is not int or not 0 <= priority <= 255):
            raise ValueError('priority must be an integer in 0..255.')
        if bpu_cores is not None:
            if (not isinstance(bpu_cores, (list, tuple)) or not bpu_cores
                    or any(type(core) is not int or core < 0 for core in bpu_cores)):
                raise ValueError('bpu-cores must be nonempty nonnegative integer indexes.')
            if self.selection.target == 'x5':
                raise ValueError('X5 does not expose BPU core selection in this sample.')
        self.load()
        if priority is None and bpu_cores is None:
            return
        for adapter in (self.encoder, self.decoder):
            binding = adapter.binding
            try:
                kwargs = {}
                if priority is not None:
                    # The native API takes per-model Mappings on every target; a scalar
                    # is an incompatible-arguments TypeError on the real X5 SDK
                    # (board evidence 2026-09-24), so never send one.
                    kwargs['priority'] = {binding.model_name: priority}
                if bpu_cores is not None:
                    kwargs['bpu_cores'] = {binding.model_name: list(bpu_cores)}
                adapter.runtime.set_scheduling_params(**kwargs)
            except Exception as exc:
                raise RuntimeError(f'{binding.stage} scheduling failed: {exc}') from exc
