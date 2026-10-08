# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""LPRNet plate recognition: load, preprocess, infer, postprocess, predict.

``LPRNetRecognizer`` owns the fixed prepacked-input contract end to end:
construction loads the model through the sample's lazy runner, and each
``predict`` call runs preprocess -> infer -> postprocess (source CTC plate
decoding) visible in this file. Catalog selection and presentation live in
``cli.py``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from utils.py_utils.assets import verify_asset_file
from utils.py_utils.platforms import require_execution_target
from utils.py_utils.runtime_meta import MetadataMismatchError, RuntimeMetadata

from samples.vision.lprnet.runtime.python.cli import ModelSelection

INPUT_SHAPE = (1, 3, 24, 94)
# Accepted logits layouts, each bound exactly as reported and matched
# strictly on every runtime call.  (1, 68, 18, 1) is the measured protocol
# of the one published X5 ``lpr.bin`` (board metadata 2026-09-24).  The 3D
# (1, 68, 18) layout is the old unified-contract/host-fixture shape kept
# for API compatibility with existing host tests and injected runners; no
# published SDK artifact has been observed reporting it.  No other rank or
# axis order is accepted — the binding never reshapes, squeezes, or
# permutes.  Singleton elimination for the CTC decoder happens only in the
# recognizer's postprocess stage.
OUTPUT_SHAPES = ((1, 68, 18), (1, 68, 18, 1))
CTC_LOGITS_SHAPE = (68, 18)

CHARS = (
    "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
    "0123456789ABCDEFGHJKLMNPQRSTUV WXYZIO-"
).replace(" ", "")
BLANK_INDEX = len(CHARS) - 1


@dataclass(frozen=True)
class ModelBinding:
    """Validated input/output names and shapes observed from one model.

    Attributes:
        selection: The manifest-backed selection this binding was built from.
        metadata: Board-observed model metadata validated against the contract.
        input_name: Bound prepacked F32 input tensor name.
        output_name: Bound native logits output tensor name.
        output_shape: Complete native logits shape reported by the metadata;
            every runtime call must reproduce it exactly.
    """

    selection: ModelSelection
    metadata: RuntimeMetadata
    input_name: str
    output_name: str
    output_shape: tuple[int, ...]

    @property
    def model_name(self) -> str:
        """Return the single submodel name declared by the artifact."""
        return self.metadata.model_name


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    """Validate source-proven LPRNet metadata and bind runtime tensor names.

    Args:
        selection: Manifest-backed selection whose asset and path must match.
        metadata: Board-observed metadata mapping or ``RuntimeMetadata``.

    Returns:
        ModelBinding: Validated tensor names and the native logits shape.

    Raises:
        BindingError: The selection does not match the exact manifest asset.
        MetadataMismatchError: Tensor names, shapes, or dtypes violate the
            contract.
    """
    from samples.vision.lprnet.runtime.python.cli import BindingError, resolve_selection

    # Do not trust a caller-created ModelSelection.  Re-resolving the manifest
    # row also protects the binding boundary from an asset-id/path substitution.
    resolved = resolve_selection(
        selection.target,
        asset_id=selection.asset.reference,
        model_path=selection.model_path if selection.explicit_model_path else None,
    )
    if (
        selection.target != resolved.target
        or selection.asset != resolved.asset
        or Path(selection.model_path) != Path(resolved.model_path)
        or selection.explicit_model_path != resolved.explicit_model_path
    ):
        raise BindingError("ModelSelection does not match the exact manifest asset and path.")

    meta = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if meta.model_names != (meta.model_name,):
        raise MetadataMismatchError("LPRNet artifact must expose exactly one model.")
    if len(meta.input_names) != 1 or len(meta.output_names) != 1:
        raise MetadataMismatchError("LPRNet requires exactly one input and one output.")
    input_name, output_name = meta.input_names[0], meta.output_names[0]
    if meta.input_shapes.get(input_name) != INPUT_SHAPE:
        raise MetadataMismatchError(f"Expected input shape {INPUT_SHAPE}, got {meta.input_shapes.get(input_name)}.")
    if meta.input_dtypes.get(input_name) != "float32":
        raise MetadataMismatchError("LPRNet input must be float32.")
    if meta.output_shapes.get(output_name) not in OUTPUT_SHAPES:
        raise MetadataMismatchError(
            f"Expected LPRNet logits metadata shape in {OUTPUT_SHAPES} "
            "(classes=68, timesteps=18; rank 4 is accepted only as "
            f"(1, 68, 18, 1)), got {meta.output_shapes.get(output_name)}."
        )
    if meta.output_dtypes.get(output_name) != "float32":
        raise MetadataMismatchError("LPRNet output must be float32.")
    return ModelBinding(
        selection, meta, input_name, output_name, meta.output_shapes[output_name]
    )


class RuntimeUnavailableError(RuntimeError):
    """The board-only hbm_runtime package is unavailable."""


class RuntimeModelRunner:
    """Load hbm_runtime only after an execution call or explicit ``load``.

    The board identity and publication-hash gates run before the SDK import
    on the real path; an injected ``runtime``/``runtime_factory`` is the
    documented host seam.
    """

    def __init__(self, selection: ModelSelection, *, runtime_factory: Callable[[str], Any] | None = None, runtime: Any = None):
        self.selection = selection
        self._runtime_factory = runtime_factory
        self._runtime = runtime
        self.binding: ModelBinding | None = None

    @property
    def loaded(self) -> bool:
        return self._runtime is not None and self.binding is not None

    @property
    def runtime(self) -> Any:
        if self._runtime is None:
            raise RuntimeError("Runtime is not loaded; call load() first.")
        return self._runtime

    def load(self) -> ModelBinding:
        if self.loaded:
            return self.binding  # type: ignore[return-value]
        if self._runtime is None:
            if self._runtime_factory is None:
                # Real execution path: the board identity and publication gates
                # run before the SDK factory is ever constructed.  Passing an
                # explicit runtime_factory is the documented host seam and is the
                # only way to skip them.
                require_execution_target(self.selection.target)
                verify_asset_file(self.selection.asset, self.selection.model_path)
            factory = self._runtime_factory or _default_runtime_factory()
            self._runtime = factory(str(self.selection.model_path))
        try:
            metadata = RuntimeMetadata.from_runtime(self._runtime)
            self.binding = bind_model(self.selection, metadata)
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
        if set(tensors) != {binding.input_name}:
            raise MetadataMismatchError(f"Expected input {binding.input_name!r}.")
        value = tensors[binding.input_name]
        if (not isinstance(value, np.ndarray) or value.shape != (1, 3, 24, 94)
                or value.dtype != np.float32 or not np.isfinite(value).all()):
            raise MetadataMismatchError("LPRNet input does not match bound shape/dtype.")
        result = self.runtime.run({binding.model_name: dict(tensors)})
        if not isinstance(result, Mapping) or set(result) != {binding.model_name}:
            raise MetadataMismatchError("Runtime returned unexpected model outputs.")
        outputs = result[binding.model_name]
        if not isinstance(outputs, Mapping) or set(outputs) != {binding.output_name}:
            raise MetadataMismatchError("Runtime returned unexpected tensor outputs.")
        raw = np.asarray(outputs[binding.output_name])
        if raw.shape != binding.output_shape or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise MetadataMismatchError(
                f"LPRNet runtime output does not match binding: expected "
                f"{binding.output_shape} float32, got {raw.shape}/{raw.dtype}."
            )
        # The raw native logits keep the bound shape (for the released
        # artifact (1, 68, 18, 1)); singleton removal belongs to postprocess.
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


def read_float32_input(path: str | Path) -> tuple[np.ndarray, Path]:
    """Read exactly one ``(1,3,24,94)`` float32 tensor and its source path.

    Args:
        path: Prepacked source fixture path; leading ~ is expanded.

    Returns:
        tuple: Owned float32 array shaped ``INPUT_SHAPE`` and the resolved
        source path recorded as this call's context.

    Raises:
        FileNotFoundError: The path does not exist.
        ValueError: The value count does not match the fixed shape.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    data = np.fromfile(source, dtype=np.float32)
    expected = int(np.prod(INPUT_SHAPE))
    if data.size != expected:
        raise ValueError(f"Expected {expected} float32 values, got {data.size}.")
    return data.reshape(INPUT_SHAPE).copy(), source


@dataclass(frozen=True)
class PreparedInput:
    """One owned tensor mapping and its per-call input context.

    Attributes:
        tensors: Input-name to prepacked F32 tensor mapping.
        context: Source fixture path of this call only.
    """

    tensors: Mapping[str, np.ndarray]
    context: Path


def ctc_logits(value: np.ndarray) -> np.ndarray:
    """Reduce bound native logits to the source ``(68, 18)`` CTC payload.

    Args:
        value: Raw bound logits array, e.g. ``(1, 68, 18, 1)``.

    Returns:
        np.ndarray: ``(68, 18)`` payload produced by singleton removal only.

    Raises:
        ValueError: The layout does not reduce by size-1 axis removal alone;
            it is rejected instead of guessed.
    """
    reduced = np.squeeze(np.asarray(value))
    if reduced.shape != CTC_LOGITS_SHAPE:
        raise ValueError(
            f"Bound logits {np.asarray(value).shape} do not reduce to "
            f"{CTC_LOGITS_SHAPE} by singleton removal; unsupported layout."
        )
    return reduced


def decode_plate(logits: np.ndarray) -> str:
    """Apply source argmax, consecutive deduplication, and blank removal.

    Args:
        logits: Squeezed ``(68, 18)`` CTC logits.

    Returns:
        str: Decoded plate text.

    Raises:
        ValueError: The logits shape is not ``(68, 18)``.
    """
    value = np.asarray(logits)
    if value.shape != (68, 18):
        raise ValueError(f"Expected squeezed logits (68, 18), got {value.shape}.")
    labels = np.argmax(value, axis=0)
    decoded: list[int] = []
    previous = int(labels[0])
    if previous != BLANK_INDEX:
        decoded.append(previous)
    for current_value in labels:
        current = int(current_value)
        if current == previous or current == BLANK_INDEX:
            if current == BLANK_INDEX:
                previous = current
            continue
        decoded.append(current)
        previous = current
    return "".join(CHARS[index] for index in decoded)


class LPRNetRecognizer:
    """Recognize one license plate with a compiled LPRNet model.

    ``predict`` composes ``preprocess`` → ``infer`` → ``postprocess``; the
    established ``pre_process``/``forward``/``post_process`` names stay thin
    aliases of those implementations.

    Attributes:
        runner (RuntimeModelRunner): Lazy sample runner used by infer.
        binding (ModelBinding): Validated tensor names and native shape.
    """

    def __init__(self, selection: ModelSelection, *, runner=None) -> None:
        """Load the compiled model and validate its tensor protocol.

        Args:
            selection: Manifest-backed selection from ``cli.resolve_selection``.
            runner: Optional injected runner (host-test seam); defaults to
                the lazy sample runner with the board-identity and
                publication-hash gates.

        Returns:
            None.

        Raises:
            ValueError: The selection or its local file is invalid.
            MetadataMismatchError: Runtime metadata violates the contract.
            RuntimeError: Board identity or SDK loading fails.
        """
        self.runner = runner if runner is not None else RuntimeModelRunner(selection)
        self.binding = self.runner.load()

    def preprocess(self, test_bin: str | Path) -> PreparedInput:
        """Read the source-provided float32 binary input without image transforms.

        Args:
            test_bin: Prepacked ``.dat`` fixture path.

        Returns:
            PreparedInput: Input-name mapping of one owned float32
            (1, 3, 24, 94) tensor plus this call's source path.

        Raises:
            FileNotFoundError: The fixture path does not exist.
            ValueError: The value count does not match the fixed shape.
        """
        tensor, path = read_float32_input(test_bin)
        return PreparedInput({self.binding.input_name: tensor}, path)

    def infer(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Run the selected model and return an owned raw float32 logits array.

        Args:
            tensors: Input mapping returned by ``preprocess(...).tensors``.

        Returns:
            np.ndarray: Raw owned float32 logits in the bound native shape;
            singleton removal belongs to postprocess.

        Raises:
            MetadataMismatchError: Input or output violates the binding.
            RuntimeError: SDK execution fails.
        """
        return self.runner(tensors)

    def postprocess(self, raw: np.ndarray) -> str:
        """Drop protocol singletons from the bound native logits and decode.

        Args:
            raw: Output of ``infer``; must be the float32 array exactly as
                bound (the released artifact reports ``(1, 68, 18, 1)``).

        Returns:
            str: Plate text from source argmax, dedup, and blank removal.

        Raises:
            ValueError: Shape or dtype does not match the bound contract.
        """
        value = np.asarray(raw)
        if value.shape != self.binding.output_shape or value.dtype != np.float32:
            raise ValueError(
                f"Expected raw float32 logits {self.binding.output_shape}, "
                f"got {value.shape}/{value.dtype}."
            )
        return decode_plate(ctc_logits(value))

    def predict(self, test_bin: str | Path) -> str:
        """Run preprocess, infer, and postprocess for one binary input.

        Args:
            test_bin: Prepacked ``.dat`` fixture path.

        Returns:
            str: Recognized plate text; see postprocess.

        Raises:
            FileNotFoundError: The fixture path does not exist.
            ValueError: Tensor data or logits are invalid.
            MetadataMismatchError: Tensor structure violates the binding.
            RuntimeError: SDK execution fails.
        """
        prepared = self.preprocess(test_bin)
        return self.postprocess(self.infer(prepared.tensors))

    def set_scheduling_params(self, *, priority=None, bpu_cores=None) -> None:
        """Apply scheduling options to the loaded board runtime.

        Args:
            priority: Optional integer in [0, 255]; None leaves it unchanged.
            bpu_cores: Optional non-empty list of nonnegative BPU core indexes.

        Returns:
            None.

        Raises:
            ValueError: Priority or a core index is out of range.
            RuntimeError: The SDK cannot apply the scheduling options.
        """
        self.runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

    def pre_process(self, test_bin: str | Path) -> PreparedInput:
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(test_bin)

    def forward(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Compatibility alias for :meth:`infer`."""
        return self.infer(tensors)

    def post_process(self, raw: np.ndarray) -> str:
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(raw)

    def __call__(self, test_bin: str | Path) -> str:
        """Delegate to predict with the same input and error contract."""
        return self.predict(test_bin)


__all__ = [
    "BLANK_INDEX", "CHARS", "LPRNetRecognizer", "PreparedInput",
    "RuntimeModelRunner", "ctc_logits", "decode_plate", "read_float32_input",
]
