# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Physical backend for the PaddleOCR two-stage pipeline.

``backend.py`` owns the coherent SDK surface: the per-stage tensor contracts
and their binding to observed metadata, the lazy stage runners with
scheduling validation, and the default runtime factory. Published pair
identity/listing lives in ``cli.py``; the readable two-stage composition and
its decode/geometry/input math live in ``ocr.py``.
"""

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

from utils.py_utils.runtime_meta import RuntimeMetadata, canonicalise_dtype

from samples.vision.paddle_ocr.runtime.python.cli import (
    BindingError,
    StageContract,
    MetadataMismatchError,
    OCRPair,
    UnsupportedAssetError,
    resolve_pair,
)

# ======================================================================
# Per-stage tensor contracts and metadata binding.
# ======================================================================



@dataclass(frozen=True)
class StageBinding:
    """Static contract plus the exact metadata observed at runtime."""

    pair: OCRPair
    stage: Literal["detector", "recognizer"]
    contract: StageContract
    model_name: str
    input_names: tuple[str, ...]
    input_shapes: Mapping[str, tuple[int, ...]]
    input_dtypes: Mapping[str, str]
    output_name: str
    output_shape: tuple[int, ...]
    output_dtype: str
    runtime_metadata: RuntimeMetadata

    @property
    def runtime_input_shapes(self) -> Mapping[str, tuple[int, ...]]:
        """Return the physical tensor shapes used by the runtime runner."""

        return self.contract.runtime_input_shapes

    @property
    def runtime_input_dtypes(self) -> Mapping[str, str]:
        """Return the physical tensor dtypes used by the runtime runner."""

        return self.contract.runtime_input_dtypes


def bind_stage(
    pair: OCRPair,
    stage: Literal["detector", "recognizer"] | str,
    metadata: RuntimeMetadata | Mapping[str, Any],
) -> StageBinding:
    """Validate actual runtime metadata against the selected stage contract."""

    if stage not in ("detector", "recognizer"):
        raise BindingError(f"Unknown OCR stage {stage!r}.")
    contract = pair.detector if stage == "detector" else pair.recognizer
    facts = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)

    if facts.model_name != contract.model_name:
        raise MetadataMismatchError(
            f"{stage} model name {facts.model_name!r} does not match the audited "
            f"name {contract.model_name!r}."
        )
    if tuple(facts.input_names) != tuple(contract.input_names):
        raise MetadataMismatchError(
            f"{stage} input names {facts.input_names!r} do not match the audited "
            f"names {contract.input_names!r}."
        )
    if tuple(facts.output_names) != (contract.output_name,):
        raise MetadataMismatchError(
            f"{stage} output names {facts.output_names!r} do not match the audited "
            f"name {contract.output_name!r}."
        )
    for name in contract.input_names:
        actual_shape = facts.input_shapes.get(name)
        if tuple(actual_shape or ()) != tuple(contract.input_shapes[name]):
            raise MetadataMismatchError(
                f"{stage} input {name!r} shape {actual_shape!r} does not match "
                f"{contract.input_shapes[name]!r}."
            )
        actual_dtype = canonicalise_dtype(facts.input_dtypes.get(name))
        if actual_dtype is None:
            raise MetadataMismatchError(
                f"{stage} input {name!r} metadata is missing a dtype."
            )
        if actual_dtype != contract.input_dtypes[name]:
            raise MetadataMismatchError(
                f"{stage} input {name!r} dtype {actual_dtype!r} does not match "
                f"the audited {contract.input_dtypes[name]!r} contract."
            )

    actual_output_shape = facts.output_shapes.get(contract.output_name)
    if tuple(actual_output_shape or ()) != tuple(contract.output_shape):
        raise MetadataMismatchError(
            f"{stage} output {contract.output_name!r} shape {actual_output_shape!r} "
            f"does not match {contract.output_shape!r}."
        )
    actual_output_dtype = canonicalise_dtype(facts.output_dtypes.get(contract.output_name))
    if actual_output_dtype is None:
        raise MetadataMismatchError(
            f"{stage} output {contract.output_name!r} metadata is missing a dtype."
        )
    if actual_output_dtype != "float32":
        raise MetadataMismatchError(
            f"The OCR pilot accepts only F32 outputs; {stage} reported "
            f"{actual_output_dtype!r}."
        )
    # SDK metadata may retain quantization descriptors for an F32 output.
    # The shared raw_f32 contract uses the actual tensor dtype; applying a
    # second dequantization here would change already decoded OCR scores.

    return StageBinding(
        pair=pair,
        stage=stage,  # type: ignore[arg-type]
        contract=contract,
        model_name=facts.model_name,
        input_names=facts.input_names,
        input_shapes=facts.input_shapes,
        input_dtypes=facts.input_dtypes,
        output_name=contract.output_name,
        output_shape=contract.output_shape,
        output_dtype=actual_output_dtype,
        runtime_metadata=facts,
    )


def validate_stage_inputs(
    binding: StageBinding | StageContract,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return a copy of one flat physical input mapping."""

    expected_names = tuple(binding.input_names)
    actual_names = tuple(inputs.keys())
    if set(actual_names) != set(expected_names) or len(actual_names) != len(expected_names):
        raise MetadataMismatchError(
            f"{binding.stage} runner inputs {actual_names!r} do not match "
            f"{expected_names!r}."
        )
    import numpy as np

    result: dict[str, Any] = {}
    for name in expected_names:
        value = np.asarray(inputs[name])
        expected_shape = binding.runtime_input_shapes[name]
        expected_dtype = np.dtype(binding.runtime_input_dtypes[name])
        if tuple(value.shape) != tuple(expected_shape):
            raise MetadataMismatchError(
                f"{binding.stage} input {name!r} runtime shape {value.shape!r} "
                f"does not match {expected_shape!r}."
            )
        if value.dtype != expected_dtype:
            raise MetadataMismatchError(
                f"{binding.stage} input {name!r} dtype {value.dtype!r} does not "
                f"match {expected_dtype!r}."
            )
        if not np.all(np.isfinite(value)):
            raise MetadataMismatchError(
                f"{binding.stage} input {name!r} contains NaN or infinity."
            )
        result[name] = value
    return result


def validate_stage_output(
    binding: StageBinding | StageContract,
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one flat F32 output mapping, including finite values."""

    import numpy as np

    if tuple(outputs.keys()) != (binding.output_name,):
        raise MetadataMismatchError(
            f"{binding.stage} runner outputs {tuple(outputs.keys())!r} do not "
            f"match {(binding.output_name,)!r}."
        )
    value = np.asarray(outputs[binding.output_name])
    if tuple(value.shape) != tuple(binding.output_shape):
        raise MetadataMismatchError(
            f"{binding.stage} output {binding.output_name!r} runtime shape "
            f"{value.shape!r} does not match {binding.output_shape!r}."
        )
    if value.dtype != np.dtype("float32"):
        raise MetadataMismatchError(
            f"{binding.stage} output {binding.output_name!r} dtype {value.dtype!r} "
            "does not match the F32 contract."
        )
    if not np.all(np.isfinite(value)):
        raise MetadataMismatchError(
            f"{binding.stage} output {binding.output_name!r} contains NaN or infinity."
        )
    return {binding.output_name: value}

# ======================================================================
# Lazy, metadata-bound stage runners.
# ======================================================================

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



import importlib
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from samples.vision.paddle_ocr.runtime.python.backend import (
    RuntimeMetadata,
    StageBinding,
    bind_stage,
    validate_stage_inputs,
    validate_stage_output,
)
from samples.vision.paddle_ocr.runtime.python.cli import MetadataMismatchError, OCRPair


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
            from utils.py_utils.platforms import require_execution_target

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
