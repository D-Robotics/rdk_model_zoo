"""Lazy board-runtime runner for a validated classification model binding.

The constructor is SDK-free.  ``runtime_factory`` and ``runtime`` are
explicit injection seams for host tests; production callers leave both unset
and use the installed board runtime.  The production load path goes through
:class:`utils.py_utils.runtime.RuntimeSession`, which owns the exact-target
identity gate, the SDK import and the model construction; execution then
continues through this runner's validated call on the loaded SDK object —
behaviorally identical to the session's ``run`` passthrough, without a
second execution path.  ``RuntimeUnavailableError`` is the session's own
exception type, re-exported here for established callers.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np

from utils.py_utils.cls_binding import (
    ClassificationContract,
    SCORE_POLICIES,
    SUPPORTED_TARGETS,
    VariantFacts,
    MetadataMismatchError,
    ModelBinding,
    ModelSelection,
    RuntimeMetadata,
    SampleBindingTable,
    bind_model,
    score_vector_shape,
)
from utils.py_utils.quantization import validate_output_transform
from utils.py_utils.runtime import RuntimeSession
from utils.py_utils.runtime import RuntimeUnavailableError  # noqa: F401 - re-exported surface
from utils.py_utils.runtime import _default_runtime_factory  # noqa: F401 - re-exported seam
from utils.py_utils.tensor_io import validate_input_tensors


class RuntimeModelRunner:
    """Load and execute one classification model through a validated binding.

    Construction accepts a ModelSelection and optional binding/SDK providers;
    see __init__ for parameter definitions. The SDK is loaded by load or the
    first call, and preprocessing and output activation remain with the caller.

    Attributes:
        selection (ModelSelection): Artifact path, target, and tensor contract.
        binding (ModelBinding | None): Validated tensor binding after load.
        metadata (RuntimeMetadata | None): SDK metadata snapshot after load.

    Notes:
        Concurrent calls on one loaded SDK instance are not guaranteed safe.
    """

    def __init__(
        self,
        selection: ModelSelection,
        *,
        table: Optional[SampleBindingTable] = None,
        binding_loader: Optional[Callable[..., ModelBinding]] = None,
        runtime_factory: Optional[Callable[[str], Any]] = None,
        runtime: Any = None,
    ) -> None:
        """Store a model selection and providers without loading the SDK.

        Args:
            selection: Concrete artifact path, target, and classification contract.
            table: Published-artifact table; may be omitted for custom selections
                or when a binding_loader is supplied.
            binding_loader: Optional callable accepting (selection, metadata) and
                returning a validated ModelBinding instead of the shared binder.
            runtime_factory: Optional callable accepting a model path and returning
                an SDK object. Injected factories bypass board identity checks.
            runtime: Optional prebuilt SDK object; takes precedence over the factory
                and bypasses board identity checks.

        Returns:
            None.
        """
        self.selection = selection
        self._table = table
        self._binding_loader = binding_loader
        self._runtime_factory = runtime_factory
        self._runtime = runtime
        self.binding: Optional[ModelBinding] = None
        self.metadata: Optional[RuntimeMetadata] = None

    @classmethod
    def from_file(
        cls, model_path: str | Path, *, target: str,
        input_size: tuple[int, int], class_count: int,
        resize_type: int = 1, resize_interpolation: str = "linear",
        score_policy: str = "softmax", output_transform: str = "raw_f32",
        resize_shorter: int = 0,
        runtime_factory: Optional[Callable[[str], Any]] = None, runtime: Any = None,
    ) -> RuntimeModelRunner:
        """Prepare a lazy NV12 classification runner from a local artifact.

        Args:
            model_path: Compiled model path; leading ~ is expanded.
            target: Concrete artifact target: x5, s100, s100p, or s600.
            input_size: Positive, even (height, width) of the model input in pixels.
            class_count: Positive number of output classes.
            resize_type: 0 stretches, 1 letterboxes (default), 2 resizes the shorter
                edge to resize_shorter and center-crops (needs a square input and Pillow).
            resize_interpolation: Direct-resize interpolation name; defaults to linear.
                Image preprocessing validates it when called.
            score_policy: softmax, legacy_softmax, or none; defaults to softmax.
            output_transform: raw_f32 or dequant; defaults to raw_f32.
            resize_shorter: Shorter-edge size for resize_type 2, at least the input
                size; 0 (default) for the other resize types.
            runtime_factory: Optional model-path-to-SDK-object factory for injection.
                When supplied, file and hardware checks are bypassed.
            runtime: Optional prebuilt SDK object; overrides runtime_factory and
                bypasses file and hardware checks.

        Returns:
            RuntimeModelRunner: Unloaded runner with a local classification contract.
            X5 uses packed NV12; S uses split Y/UV. Call load to bind SDK metadata.

        Raises:
            ValueError: Target, dimensions, class count, resize type, or score policy
                is invalid.
            OutputTransformError: The output transform name is unsupported.
            FileNotFoundError: The artifact is missing and no SDK object/factory is injected.

        Notes:
            Construction does not import the SDK. Production load validates board
            identity and tensor metadata. This method does not access the model catalog.
        """
        height, width = input_size
        if target not in SUPPORTED_TARGETS:
            raise ValueError(f"target must be one of {SUPPORTED_TARGETS}.")
        if height <= 0 or width <= 0 or height % 2 or width % 2:
            raise ValueError("NV12 input dimensions must be positive and even.")
        if class_count <= 0:
            raise ValueError("class_count must be positive.")
        if resize_type not in (0, 1, 2) or score_policy not in SCORE_POLICIES:
            raise ValueError("Invalid resize_type or score_policy.")
        if resize_type == 2 and (height != width or resize_shorter < width):
            raise ValueError(
                "resize_type 2 needs a square input and resize_shorter >= the input size.")
        path = Path(model_path).expanduser()
        if runtime is None and runtime_factory is None and not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        facts = VariantFacts(
            input_height=height, input_width=width, class_count=class_count,
            resize_type=resize_type, resize_interpolation=resize_interpolation,
            resize_shorter=resize_shorter, output_score_policy=score_policy,
            output_transform=validate_output_transform(output_transform))
        contract = ClassificationContract(
            asset_id=f"local:{path.name}", variant="custom", target=target,
            model_format=path.suffix.lstrip("."), source_manifest="(local model)",
            input_protocol="packed_nv12" if target == "x5" else "split_nv12",
            **asdict(facts))
        selection = ModelSelection(
            contract.asset_id, "custom", target, path, contract, "local",
            explicit_model_path=True, custom=True)
        return cls(selection, runtime_factory=runtime_factory, runtime=runtime)

    @property
    def loaded(self) -> bool:
        """Report whether both SDK construction and metadata binding succeeded.

        Returns:
            bool: True when runtime and binding are available; otherwise False.
        """

        return self._runtime is not None and self.binding is not None

    @property
    def runtime(self) -> Any:
        """Return the current SDK object without triggering a load.

        Returns:
            Any: SDK object supplied at construction or created by load.

        Raises:
            RuntimeError: No SDK object is available.

        Notes:
            Use loaded to check that tensor metadata has also been validated.
        """

        if self._runtime is None:
            raise RuntimeError("Runtime has not been loaded; call load() first.")
        return self._runtime

    def load(self) -> ModelBinding:
        """Load the SDK model and validate its tensor metadata once.

        Returns:
            ModelBinding: Validated classification binding; reused on later calls.

        Raises:
            ValueError: A published selection lacks a table or binding loader, or
                the requested target cannot execute on this board.
            BindingError: Artifact identity or SDK metadata violates the contract.
            RuntimeError: The SDK is unavailable or cannot construct the model.

        Notes:
            Production loading verifies board identity before importing the SDK.
            An injected SDK object/factory bypasses that check. Failed metadata
            validation clears runtime, metadata, and binding so loading can be retried.
        """

        if self.loaded:
            return self.binding  # type: ignore[return-value]

        # Importing the shared platform module is dependency-free.  The check
        # is deferred to execution so help/list/dry-run and host imports stay
        # usable without a board.
        if self._runtime is None and self._runtime_factory is None:
            # Production path: the shared session owns the exact-target
            # identity gate, the SDK import and the model construction.
            session = RuntimeSession(
                str(self.selection.model_path), target=self.selection.target)
            try:
                session.load()
                self._runtime = session.runtime
            except Exception:
                # Preserve the original runtime exception and avoid pretending
                # that a failed model construction was a valid empty runner.
                self._runtime = None
                raise
        elif self._runtime is None:
            # ``runtime_factory`` stays the documented host-test seam and
            # injects construction directly; it carries no claim about local
            # hardware, so the identity gate is not applied to it.
            try:
                self._runtime = self._runtime_factory(
                    str(self.selection.model_path))
            except Exception:
                self._runtime = None
                raise
        try:
            self.metadata = RuntimeMetadata.from_runtime(self._runtime)
            if self._binding_loader is not None:
                self.binding = self._binding_loader(self.selection, self.metadata)
            else:
                if self._table is None and not self.selection.custom:
                    raise ValueError(
                        "RuntimeModelRunner needs the sample binding table (or "
                        "a binding_loader) to validate runtime metadata."
                    )
                self.binding = bind_model(self._table, self.selection, self.metadata)
        except Exception:
            self._runtime = None
            self.metadata = None
            self.binding = None
            raise
        return self.binding

    def set_scheduling_params(self, *, priority: Optional[int] = None,
                              bpu_cores: Optional[list[int]] = None) -> None:
        """Load the model and pass scheduling options to its SDK instance.

        Args:
            priority: Optional integer priority in [0, 255]; None leaves it unchanged.
            bpu_cores: Optional list of nonnegative core indexes. The SDK validates
                hardware-specific availability; None leaves core selection unchanged.

        Returns:
            None.

        Raises:
            ValueError: Priority or a core index is out of range.
            RuntimeError: Loading fails or the SDK cannot apply scheduling parameters.

        Notes:
            Even with both arguments None, this method ensures the model is loaded.
        """

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
        """Validate input tensors and execute one SDK inference call.

        Args:
            inputs: Tensor-name mapping of contiguous uint8 NV12 arrays with values
                in [0, 255]. At model size (H, W), packed input is (H*W*3//2,);
                split input is Y (1, H, W, 1) plus UV (1, H//2, W//2, 2).

        Returns:
            Mapping[str, np.ndarray]: One raw score tensor keyed by bound output name,
            with a shape that squeezes to (class_count,). raw_f32 requires float32;
            dequant retains the SDK dtype. No activation or dequantization is applied.

        Raises:
            ValueError: Input names, shapes, dtype, or contiguity violate the contract.
            MetadataMismatchError: Output container, shape, or raw_f32 dtype is invalid.
            RuntimeError: Model loading or SDK execution fails.
        """

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
                  table: Optional[SampleBindingTable] = None,
                  binding_loader: Optional[Callable[..., ModelBinding]] = None,
                  runtime_factory: Optional[Callable[[str], Any]] = None,
                  runtime: Any = None) -> RuntimeModelRunner:
    """Construct an unloaded runner for a resolved classification selection.

    Args:
        selection: Concrete artifact path, target, and classification contract.
        table: Optional published-artifact table passed to RuntimeModelRunner.
        binding_loader: Optional (selection, metadata) to ModelBinding callable.
        runtime_factory: Optional model-path-to-SDK-object factory for injection.
        runtime: Optional prebuilt SDK object; takes precedence over the factory.

    Returns:
        RuntimeModelRunner: Lazy runner with no SDK loading performed here.
    """

    return RuntimeModelRunner(
        selection,
        table=table,
        binding_loader=binding_loader,
        runtime_factory=runtime_factory,
        runtime=runtime,
    )


__all__ = ["RuntimeModelRunner", "RuntimeUnavailableError", "create_runner"]
