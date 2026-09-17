# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy, metadata-bound runtime runners for the two OCR stages."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from samples.vision.paddle_ocr.runtime.python.model_binding import (
    MetadataMismatchError,
    OCRPair,
    RuntimeMetadata,
    StageBinding,
    bind_stage,
    validate_stage_inputs,
    validate_stage_output,
)


class RuntimeUnavailableError(RuntimeError):
    """The board runtime is unavailable in the current Python environment."""


class RuntimeStageRunner:
    """Load one selected OCR stage only when it is first executed.

    The constructor is SDK-free.  ``runtime`` and ``runtime_factory`` are
    explicit host-test seams; production callers leave them unset and use the
    matching target's installed ``hbm_runtime`` package.
    """

    def __init__(
        self,
        pair: OCRPair,
        stage: str,
        *,
        runtime_factory: Callable[[str], Any] | None = None,
        runtime: Any = None,
        priority: int | None = None,
        bpu_cores: list[int] | None = None,
    ) -> None:
        if stage not in ("detector", "recognizer"):
            raise ValueError(f"Unknown OCR stage {stage!r}.")
        if runtime is not None and runtime_factory is not None:
            raise ValueError("Provide either runtime or runtime_factory, not both.")
        self.pair = pair
        self.stage = stage
        self._runtime_factory = runtime_factory
        self._runtime = runtime
        self.binding: StageBinding | None = None
        self.metadata: RuntimeMetadata | None = None
        self._priority = _validate_priority(priority)
        self._bpu_cores = _validate_cores(bpu_cores)

    @property
    def loaded(self) -> bool:
        """Whether runtime construction and metadata binding succeeded."""

        return self._runtime is not None and self.binding is not None

    @property
    def runtime(self) -> Any:
        """Return the loaded SDK object for legacy compatibility adapters.

        New code should call the runner itself.  The property exists so the
        old per-platform wrappers can expose their historical ``model``
        attribute without constructing a second ``HB_HBMRuntime`` instance.
        """

        self.load()
        if self._runtime is None:  # pragma: no cover - load() raises first
            raise RuntimeError(f"{self.stage} runtime is not loaded.")
        return self._runtime

    @property
    def model_path(self):
        """Return the selected local model path for diagnostics."""

        return (
            self.pair.detector_model_path
            if self.stage == "detector"
            else self.pair.recognizer_model_path
        )

    def load(self) -> StageBinding:
        """Authorize the target, load the model, and validate actual metadata."""

        if self.loaded:
            return self.binding  # type: ignore[return-value]

        # An injected runtime/factory is a host-test seam and carries no claim
        # about the current machine.  Production construction checks the exact
        # execution target immediately before importing/constructing the SDK.
        if self._runtime is None and self._runtime_factory is None:
            from samples._shared.platforms import require_execution_target

            require_execution_target(self.pair.target)

        if self._runtime is None:
            factory = self._runtime_factory or _default_runtime_factory()
            try:
                self._runtime = factory(str(self.model_path))
            except Exception:
                self._runtime = None
                raise

        try:
            self.metadata = RuntimeMetadata.from_runtime(self._runtime)
            contract = (
                self.pair.detector
                if self.stage == "detector"
                else self.pair.recognizer
            )
            self.binding = bind_stage(self.pair, self.stage, self.metadata)
            if self.binding.contract != contract:  # defensive invariant
                raise MetadataMismatchError("Runtime stage binding selected the wrong contract.")
            self._apply_scheduling()
        except Exception:
            # Do not retain a partially bound runtime after metadata rejection.
            self._runtime = None
            self.metadata = None
            self.binding = None
            raise
        return self.binding

    def set_scheduling_params(
        self,
        *,
        priority: int | None = None,
        bpu_cores: list[int] | None = None,
    ) -> None:
        """Store or apply board scheduling options through the SDK runtime."""

        self._priority = _validate_priority(priority)
        self._bpu_cores = _validate_cores(bpu_cores)
        if self.loaded:
            self._apply_scheduling()

    def __call__(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Validate flat physical inputs, run the SDK, and return flat outputs."""

        binding = self.load()
        prepared = validate_stage_inputs(binding, inputs)
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError(f"{self.stage} runtime is not loaded.")
        try:
            outputs = runtime.run({binding.model_name: prepared})
        except Exception as exc:
            raise RuntimeError(f"{self.stage} runtime call failed: {exc}") from exc
        if not isinstance(outputs, Mapping):
            raise MetadataMismatchError(
                f"{self.stage} runtime returned a non-mapping output."
            )
        flat = outputs.get(binding.model_name, outputs)
        if not isinstance(flat, Mapping):
            raise MetadataMismatchError(
                f"{self.stage} runtime output is not a flat tensor mapping."
            )
        return validate_stage_output(binding, flat)

    def _apply_scheduling(self) -> None:
        if self._priority is None and self._bpu_cores is None:
            return
        runtime = self._runtime
        binding = self.binding
        if runtime is None or binding is None:
            raise RuntimeError(f"{self.stage} runtime is not loaded.")
        setter = getattr(runtime, "set_scheduling_params", None)
        if not callable(setter):
            raise RuntimeError(
                "The installed OCR runtime does not expose scheduling parameters."
            )
        kwargs: dict[str, Any] = {}
        if self._priority is not None:
            kwargs["priority"] = {binding.model_name: self._priority}
        if self._bpu_cores is not None:
            kwargs["bpu_cores"] = {binding.model_name: list(self._bpu_cores)}
        setter(**kwargs)


def create_stage_runners(
    pair: OCRPair,
    *,
    priority: int | None = None,
    bpu_cores: list[int] | None = None,
    detector_runtime_factory: Callable[[str], Any] | None = None,
    recognizer_runtime_factory: Callable[[str], Any] | None = None,
    detector_runtime: Any = None,
    recognizer_runtime: Any = None,
) -> tuple[RuntimeStageRunner, RuntimeStageRunner]:
    """Construct independently injectable lazy detector and recognizer runners."""

    return (
        RuntimeStageRunner(
            pair,
            "detector",
            runtime_factory=detector_runtime_factory,
            runtime=detector_runtime,
            priority=priority,
            bpu_cores=bpu_cores,
        ),
        RuntimeStageRunner(
            pair,
            "recognizer",
            runtime_factory=recognizer_runtime_factory,
            runtime=recognizer_runtime,
            priority=priority,
            bpu_cores=bpu_cores,
        ),
    )


def _validate_priority(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        raise ValueError("priority must be an integer between 0 and 255.")
    return value


def _validate_cores(values: list[int] | None) -> list[int] | None:
    if values is None:
        return None
    result = list(values)
    if any(isinstance(core, bool) or not isinstance(core, int) or core < 0 for core in result):
        raise ValueError("bpu_cores must contain non-negative integer indexes.")
    return result


def _default_runtime_factory() -> Callable[[str], Any]:
    try:
        runtime_module = importlib.import_module("hbm_runtime")
    except ImportError as exc:  # pragma: no cover - board environment only
        raise RuntimeUnavailableError(
            "hbm_runtime is required for OCR board execution. Use the matching "
            "RDK Python environment; help/list/dry-run do not need it."
        ) from exc
    factory = getattr(runtime_module, "HB_HBMRuntime", None)
    if not callable(factory):
        raise RuntimeUnavailableError(
            "Installed hbm_runtime does not expose HB_HBMRuntime."
        )
    return factory


__all__ = [
    "RuntimeStageRunner",
    "RuntimeUnavailableError",
    "create_stage_runners",
]
