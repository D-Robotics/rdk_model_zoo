# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lazy runtime adapter for the five-dimensional R3D-18 contract."""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from samples._shared.model_runner import _default_runtime_factory
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata
from .model_binding import ModelBinding, ModelSelection, bind_model


class RuntimeModelRunner:
    def __init__(
        self,
        selection: ModelSelection,
        *,
        runtime: Any = None,
        runtime_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.selection = selection
        self._runtime = runtime
        self._runtime_factory = runtime_factory
        self.metadata: RuntimeMetadata | None = None
        self.binding: ModelBinding | None = None

    @property
    def loaded(self) -> bool:
        return self._runtime is not None and self.binding is not None

    @property
    def runtime(self) -> Any:
        if self._runtime is None:
            raise RuntimeError("Runtime has not been loaded; call load() first.")
        return self._runtime

    def load(self) -> ModelBinding:
        if self.loaded:
            return self.binding  # type: ignore[return-value]
        if self._runtime is None and self._runtime_factory is None:
            from samples._shared.platforms import require_execution_target

            require_execution_target(self.selection.target)
        try:
            if self._runtime is None:
                factory = self._runtime_factory or _default_runtime_factory()
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
        binding = self.load()
        if priority is not None and not 0 <= priority <= 255:
            raise ValueError("priority must be between 0 and 255.")
        if bpu_cores is not None and (not bpu_cores or any(core < 0 for core in bpu_cores)):
            raise ValueError("bpu_cores must be a nonempty list of nonnegative indexes.")
        if priority is None and bpu_cores is None:
            return
        setter = getattr(self.runtime, "set_scheduling_params", None)
        if not callable(setter):
            raise RuntimeError("The installed runtime does not expose scheduling parameters.")
        kwargs: dict[str, Any] = {}
        if priority is not None:
            kwargs["priority"] = {binding.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {binding.model_name: list(bpu_cores)}
        setter(**kwargs)

    def __call__(self, tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        binding = self.load()
        if not isinstance(tensors, Mapping) or set(tensors) != {binding.input_name}:
            raise MetadataMismatchError(f"R3D-18 requires exactly one input named {binding.input_name!r}.")
        value = np.asarray(tensors[binding.input_name])
        if (
            value.shape != binding.input_shape
            or value.dtype != np.dtype(binding.input_dtype)
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
        ):
            raise MetadataMismatchError("R3D-18 input must be finite contiguous float32 (1,3,16,112,112).")
        outputs = self.runtime.run({binding.model_name: dict(tensors)})
        if not isinstance(outputs, Mapping):
            raise MetadataMismatchError("Runtime returned a non-mapping output.")
        flat = outputs.get(binding.model_name, outputs)
        if not isinstance(flat, Mapping) or set(flat) != {binding.output_name}:
            raise MetadataMismatchError("Runtime output must contain exactly the bound score tensor.")
        output = np.asarray(flat[binding.output_name])
        if output.shape != binding.output_shape or output.dtype != np.dtype(binding.output_dtype):
            raise MetadataMismatchError(
                f"Runtime output {binding.output_name!r} differs from bound shape/dtype."
            )
        if not np.isfinite(output).all():
            raise MetadataMismatchError("R3D-18 output contains NaN or infinity.")
        return {binding.output_name: output}


__all__ = ["RuntimeModelRunner"]
