# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lazy transport for named physical inputs and one raw output array.

Task bindings retain artifact identity, tensor geometry and numeric semantics.
This module owns only SDK loading, scheduling, structural validation and an
owned raw output. It never normalizes, dequantizes, activates or decodes arrays.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping

import numpy as np

from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata
from samples._shared.assets import verify_asset_file
from samples._shared.platforms import require_execution_target


class RuntimeUnavailableError(RuntimeError):
    """The board-only hbm_runtime package is unavailable."""


class SingleArrayRunner:
    """Load one bound model lazily; logical and physical inputs may differ."""

    def __init__(
        self,
        selection: Any,
        *,
        binding_loader: Callable[[Any, RuntimeMetadata], Any],
        physical_input: Callable[[Any], tuple[tuple[int, ...], Any]] | None = None,
        physical_inputs: Callable[[Any], Mapping[str, tuple[tuple[int, ...], Any]]] | None = None,
        task_name: str,
        runtime_factory: Callable[[str], Any] | None = None,
        runtime: Any = None,
        execution_target_gate: Callable[[str], Any] = require_execution_target,
    ):
        self.selection = selection
        self._binding_loader = binding_loader
        if (physical_input is None) == (physical_inputs is None):
            raise ValueError("Provide exactly one physical_input or physical_inputs contract")
        self._physical_input = physical_input
        self._physical_inputs = physical_inputs
        self._task_name = task_name
        self._execution_target_gate = execution_target_gate
        self._runtime_factory = runtime_factory
        self._runtime = runtime
        self.binding: Any = None

    @property
    def loaded(self) -> bool:
        return self._runtime is not None and self.binding is not None

    @property
    def runtime(self) -> Any:
        if self._runtime is None:
            raise RuntimeError("Runtime is not loaded; call load() first.")
        return self._runtime

    def load(self) -> Any:
        if self.loaded:
            return self.binding  # type: ignore[return-value]
        if self._runtime is None:
            if self._runtime_factory is None:
                # Real execution path: the board identity and publication gates
                # run before the SDK factory is ever constructed.  Passing an
                # explicit runtime_factory is the documented host seam and is the
                # only way to skip them.
                self._execution_target_gate(self.selection.target)
                verify_asset_file(self.selection.asset, self.selection.model_path)
            factory = self._runtime_factory or _default_runtime_factory()
            self._runtime = factory(str(self.selection.model_path))
        try:
            self.binding = self._binding_loader(self.selection, RuntimeMetadata.from_runtime(self._runtime))
        except Exception:
            self._runtime = None
            self.binding = None
            raise
        return self.binding

    def set_scheduling_params(self, *, priority: int | None = None, bpu_cores: list[int] | None = None) -> None:
        if priority is not None and (type(priority) is not int or not 0 <= priority <= 255):
            raise ValueError("priority must be between 0 and 255")
        if bpu_cores is not None and (
            not isinstance(bpu_cores, (list, tuple)) or not bpu_cores
            or any(type(core) is not int or core < 0 for core in bpu_cores)
        ):
            raise ValueError("bpu_cores must be a non-empty list of non-negative integer indexes")
        if priority is None and bpu_cores is None:
            return
        binding = self.load()
        setter = getattr(self.runtime, "set_scheduling_params", None)
        if not callable(setter):
            raise RuntimeError("Runtime does not expose set_scheduling_params")
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {binding.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {binding.model_name: bpu_cores}
        setter(**kwargs)

    def __call__(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        binding = self.load()
        expected = (
            self._physical_inputs(binding) if self._physical_inputs is not None
            else {binding.input_name: self._physical_input(binding)}
        )
        if set(tensors) != set(expected):
            raise MetadataMismatchError(f"Expected inputs {tuple(expected)!r}.")
        for name, (input_shape, input_dtype) in expected.items():
            value = tensors[name]
            if (not isinstance(value, np.ndarray) or value.shape != input_shape
                    or value.dtype != np.dtype(input_dtype) or not np.isfinite(value).all()):
                raise MetadataMismatchError(
                    f"{self._task_name} input {name!r} does not match bound shape/dtype."
                )
        result = self.runtime.run({binding.model_name: dict(tensors)})
        if not isinstance(result, Mapping) or set(result) != {binding.model_name}:
            raise MetadataMismatchError("Runtime returned unexpected model outputs.")
        outputs = result[binding.model_name]
        if not isinstance(outputs, Mapping) or set(outputs) != {binding.output_name}:
            raise MetadataMismatchError("Runtime returned unexpected tensor outputs.")
        raw = np.asarray(outputs[binding.output_name])
        if raw.shape != binding.metadata.output_shapes[binding.output_name] or raw.dtype != np.dtype(binding.metadata.output_dtypes[binding.output_name]) or not np.isfinite(raw).all():
            raise MetadataMismatchError(f"{self._task_name} runtime output does not match binding.")
        return np.array(raw, copy=True)


def _default_runtime_factory() -> Callable[[str], Any]:
    try:
        module = importlib.import_module("hbm_runtime")
    except ImportError as exc:
        raise RuntimeUnavailableError("hbm_runtime is board-only; host list/dry-run do not need it") from exc
    factory = getattr(module, "HB_HBMRuntime", None)
    if not callable(factory):
        raise RuntimeUnavailableError("hbm_runtime does not expose HB_HBMRuntime")
    return factory


__all__ = ["SingleArrayRunner", "RuntimeUnavailableError"]
