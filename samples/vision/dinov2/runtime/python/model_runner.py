# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lazy SDK adapter for the DINOv2 single-model, dual-output contract."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from samples._shared.model_runner import _default_runtime_factory
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata
from samples.vision.dinov2.runtime.python.model_binding import ModelBinding, ModelSelection, bind_model


class RuntimeModelRunner:
    """Load and validate one selected HBM lazily; instances are not thread-safe."""

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
            from samples._shared.platforms import require_execution_target

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


__all__ = ["RuntimeModelRunner"]
