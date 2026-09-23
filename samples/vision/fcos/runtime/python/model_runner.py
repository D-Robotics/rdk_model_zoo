# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Lazy X5 runtime adapter for a validated FCOS binding."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping

import numpy as np

from samples._shared.runtime_meta import RuntimeMetadata
from .model_binding import BindingError, ModelBinding, ModelSelection, bind_model


class RuntimeUnavailableError(RuntimeError):
    """The board-only ``hbm_runtime`` package is unavailable."""


class RuntimeModelRunner:
    """Load the board SDK only after an execution call requests it."""

    def __init__(self, selection: ModelSelection, *, runtime_factory: Callable[[str], Any] | None = None, runtime: Any = None):
        self.selection = selection
        self._factory = runtime_factory
        self._runtime = runtime
        self.binding: ModelBinding | None = None
        self.metadata: RuntimeMetadata | None = None

    @property
    def loaded(self) -> bool:
        """Whether runtime and binding metadata have both been loaded."""
        return self._runtime is not None and self.binding is not None

    def load(self) -> ModelBinding:
        """Load, observe, and bind one exact selected model."""
        if self.loaded:
            return self.binding  # type: ignore[return-value]
        if self._runtime is None:
            if self._factory is None:
                from samples._shared.platforms import require_execution_target
                from samples._shared.assets import resolve_asset, verify_asset_file

                require_execution_target(self.selection.target)
                verify_asset_file(resolve_asset(self.selection.asset_id), self.selection.model_path)
            factory = self._factory or _default_runtime_factory()
            self._runtime = factory(str(self.selection.model_path))
        try:
            self.metadata = RuntimeMetadata.from_runtime(self._runtime)
            self.binding = bind_model(self.selection, self.metadata)
        except Exception:
            self._runtime = None
            self.metadata = None
            self.binding = None
            raise
        return self.binding

    def set_scheduling_params(self, *, priority: int | None = None, bpu_cores: list[int] | None = None) -> None:
        """Apply source scheduling parameters after metadata binding."""
        binding = self.load()
        if priority is not None and not 0 <= priority <= 255:
            raise ValueError("priority must be between 0 and 255.")
        if bpu_cores is not None and any((not isinstance(core, int) or core < 0) for core in bpu_cores):
            raise ValueError("bpu_cores must contain non-negative integers.")
        if priority is None and bpu_cores is None:
            return
        setter = getattr(self._runtime, "set_scheduling_params", None)
        if not callable(setter):
            raise RuntimeError("hbm_runtime does not expose set_scheduling_params.")
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {binding.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {binding.model_name: bpu_cores}
        setter(**kwargs)

    def __call__(self, tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run one validated packed input and return raw output arrays."""
        binding = self.load()
        binding.validate_inputs(tensors)
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("Runtime has not been loaded.")
        outputs = runtime.run({binding.model_name: dict(tensors)})
        if not isinstance(outputs, Mapping):
            raise BindingError("Runtime returned a non-mapping output.")
        flat = outputs.get(binding.model_name, outputs)
        if not isinstance(flat, Mapping):
            raise BindingError("Runtime returned a non-mapping model output.")
        return binding.validate_outputs(flat)


def _default_runtime_factory() -> Callable[[str], Any]:
    try:
        module = importlib.import_module("hbm_runtime")
    except ImportError as exc:
        raise RuntimeUnavailableError("hbm_runtime is board-only; host help/list/dry-run do not need it.") from exc
    factory = getattr(module, "HB_HBMRuntime", None)
    if not callable(factory):
        raise RuntimeUnavailableError("hbm_runtime does not expose HB_HBMRuntime.")
    return factory


__all__ = ["RuntimeModelRunner", "RuntimeUnavailableError"]
