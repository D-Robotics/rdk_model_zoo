"""Lazy board-runtime runner for a validated ResNet model binding."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping, Optional

import numpy as np

from samples.vision.resnet.runtime.python.model_binding import (
    MetadataMismatchError,
    ModelBinding,
    ModelSelection,
    RuntimeMetadata,
    bind_model,
    score_vector_shape,
)
from samples.vision.resnet.runtime.python.tensor_io import validate_input_tensors


class RuntimeUnavailableError(RuntimeError):
    """The board runtime is unavailable in the current Python environment."""


class RuntimeModelRunner:
    """Load ``hbm_runtime`` only when the first board execution is requested.

    The constructor is SDK-free.  ``runtime_factory`` and ``runtime`` are
    explicit injection seams for host tests; production callers leave both
    unset and use the installed board runtime.
    """

    def __init__(
        self,
        selection: ModelSelection,
        *,
        runtime_factory: Optional[Callable[[str], Any]] = None,
        runtime: Any = None,
    ) -> None:
        self.selection = selection
        self._runtime_factory = runtime_factory
        self._runtime = runtime
        self.binding: Optional[ModelBinding] = None
        self.metadata: Optional[RuntimeMetadata] = None

    @property
    def loaded(self) -> bool:
        """Whether runtime construction and metadata binding both succeeded."""

        return self._runtime is not None and self.binding is not None

    @property
    def runtime(self) -> Any:
        """Return the loaded SDK object for source-compatible wrappers.

        The canonical task only needs the callable runner interface.  The
        legacy adapters also expose ``model`` because the old samples allowed
        callers to inspect the SDK object directly.  Access remains lazy: it
        raises until :meth:`load` has completed successfully.
        """

        if self._runtime is None:
            raise RuntimeError("Runtime has not been loaded; call load() first.")
        return self._runtime

    def load(self) -> ModelBinding:
        """Validate the board identity, load the model, and bind its metadata."""

        if self.loaded:
            return self.binding  # type: ignore[return-value]

        # Importing the shared platform module is dependency-free.  The check
        # is deferred to execution so help/list/dry-run and host imports stay
        # usable without a board.
        from samples._shared.platforms import require_execution_target

        # An injected runtime is a host-test seam and carries no claim about
        # local hardware.  Production construction still requires the exact
        # detected target before loading an SDK model.
        if self._runtime is None and self._runtime_factory is None:
            require_execution_target(self.selection.target)
        if self._runtime is None:
            factory = self._runtime_factory
            if factory is None:
                factory = _default_runtime_factory()
            try:
                self._runtime = factory(str(self.selection.model_path))
            except Exception:
                # Preserve the original runtime exception and avoid pretending
                # that a failed model construction was a valid empty runner.
                self._runtime = None
                raise
        try:
            self.metadata = RuntimeMetadata.from_runtime(self._runtime)
            self.binding = bind_model(self.selection, self.metadata)
        except Exception:
            self._runtime = None
            self.metadata = None
            self.binding = None
            raise
        return self.binding

    def set_scheduling_params(self, *, priority: Optional[int] = None,
                              bpu_cores: Optional[list[int]] = None) -> None:
        """Apply scheduling options through the runtime's existing API."""

        binding = self.load()
        if priority is not None and not 0 <= priority <= 255:
            raise ValueError("priority must be between 0 and 255.")
        if bpu_cores is not None and any(core < 0 for core in bpu_cores):
            raise ValueError("bpu_cores must contain non-negative indexes.")
        if priority is None and bpu_cores is None:
            return
        setter = getattr(self._runtime, "set_scheduling_params", None)
        if not callable(setter):
            raise RuntimeError("The installed runtime does not expose scheduling parameters.")
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {binding.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {binding.model_name: bpu_cores}
        setter(**kwargs)

    def __call__(self, inputs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Run one validated input mapping and return flat output tensors."""

        binding = self.load()
        validate_input_tensors(binding, inputs)
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("Runtime was not loaded.")
        outputs = runtime.run({binding.model_name: dict(inputs)})
        if not isinstance(outputs, Mapping):
            raise MetadataMismatchError("Runtime returned a non-mapping output.")
        flat = outputs.get(binding.model_name, outputs)
        if not isinstance(flat, Mapping) or binding.output_name not in flat:
            raise MetadataMismatchError(
                f"Runtime output does not contain bound tensor {binding.output_name!r}.")
        output = np.asarray(flat[binding.output_name])
        # H4: the observed shape must satisfy the squeeze rule instead of one
        # hard-coded spelling; the runner performs container validation only,
        # the declared output transform stays with post_process (H1).
        if not score_vector_shape(output.shape, binding.contract.class_count):
            raise MetadataMismatchError(
                f"Runtime output {binding.output_name!r} shape {output.shape} does "
                f"not squeeze to the bound ({binding.contract.class_count},) "
                "score vector."
            )
        if binding.output_transform == "raw_f32" and output.dtype != np.dtype("float32"):
            raise MetadataMismatchError(
                f"Runtime output {binding.output_name!r} dtype {output.dtype} does not "
                "match the declared raw_f32 contract."
            )
        return {binding.output_name: output}


def create_runner(selection: ModelSelection, *,
                  runtime_factory: Optional[Callable[[str], Any]] = None,
                  runtime: Any = None) -> RuntimeModelRunner:
    """Construct a lazy runner for one resolved selection."""

    return RuntimeModelRunner(
        selection,
        runtime_factory=runtime_factory,
        runtime=runtime,
    )


def _default_runtime_factory() -> Callable[[str], Any]:
    try:
        runtime_module = importlib.import_module("hbm_runtime")
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "hbm_runtime is required for board execution. Install/use the "
            "matching RDK Python environment; host help/list/dry-run do not need it."
        ) from exc
    factory = getattr(runtime_module, "HB_HBMRuntime", None)
    if not callable(factory):
        raise RuntimeUnavailableError(
            "Installed hbm_runtime does not expose HB_HBMRuntime."
        )
    return factory


__all__ = ["RuntimeModelRunner", "RuntimeUnavailableError", "create_runner"]
