"""Finite model and tensor contracts for the ResNet classification pilot.

The publication manifests remain the authority for asset facts.  This module
only adds the small, source-proven execution contract needed by the pilot.  It
does not import a board SDK, inspect model bytes, or infer a protocol from a
filename supplied by a caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from samples._shared.quantization import OUTPUT_TRANSFORMS, validate_output_transform
from samples._shared.runtime_meta import (
    MetadataMismatchError as _SharedMetadataMismatchError,
)
from samples._shared.runtime_meta import RuntimeMetadata, canonicalise_dtype


SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
# ResNet18 is the P1 pilot.  The existing S ResNet50/152 sources and manifest
# rows remain in their legacy locations until their compiled metadata is
# checked; they are deliberately not made executable by this pilot.
SUPPORTED_VARIANTS = ("resnet18",)
# Runtime metadata, when a future SDK exposes this optional field, may only
# report terms observed in the source/evidence.  An absent field remains the
# normal path; arbitrary or "unknown" declarations are rejected.
KNOWN_OUTPUT_SEMANTICS = frozenset({"logits", "probabilities"})
LEGACY_SCORE_POLICIES = ("legacy_softmax",)
_SAMPLE_DIR = Path(__file__).resolve().parents[2]


class BindingError(ValueError):
    """Base class for selection and runtime contract failures."""


class UnsupportedAssetError(BindingError):
    """The requested asset/target combination is not in the pilot set."""


class MetadataMismatchError(BindingError, _SharedMetadataMismatchError):
    """Runtime metadata cannot satisfy the known pilot contract."""


class ManifestAssetError(BindingError):
    """The shared publication manifest could not provide the pilot asset."""


@dataclass(frozen=True)
class ClassificationContract:
    """Known input/output facts for one published ResNet18 artifact.

    ``output_semantics`` intentionally says ``unverified_score_vector``.  The
    old wrappers apply softmax to the returned vector for both families.  The
    X5 runtime calls its tensor ``prob`` and board evidence shows a normalized
    vector, while the source and conversion documentation do not establish
    whether the graph or the wrapper owns that normalization.  The pilot keeps
    the old ``legacy_softmax`` behavior and records that uncertainty instead of
    calling the tensor a verified logits output.

    ``output_transform`` (Phase 1.5 H1) declares how raw runtime outputs
    become float32 values: the published ResNet artifacts return F32 tensors,
    so the contract declares ``raw_f32`` and the reject-int8 rule becomes a
    declared option instead of an implicit hardcode.  Output shapes follow the
    H4 rank rule (:func:`score_vector_shape`): the X5 ``(1, 1000, 1, 1)`` and
    S ``(1, 1000)`` forms are two spellings of one squeezable score vector,
    not two hard-coded contracts.
    """

    asset_id: str
    variant: str
    target: str
    model_format: str
    input_protocol: str
    input_height: int
    input_width: int
    class_count: int
    output_transform: str
    output_semantics: str
    output_score_policy: str
    resize_type: int
    resize_interpolation: str
    letterbox_interpolation: str
    source_manifest: str


@dataclass(frozen=True)
class AssetRecord:
    """Executable view of one exact shared-manifest asset row.

    ``asset_id`` is the qualified reference ``group:sample:filename``.  It is
    a UI and selection reference, not a newly invented standalone catalog ID.
    """

    asset_id: str
    variant: str
    target: str
    filename: str
    model_format: str
    source_manifest: str
    sample_id: str

    @property
    def reference(self) -> str:
        """Return the qualified manifest reference for this row."""

        return self.asset_id

    @property
    def input_protocol(self) -> str:
        """Return the physical NV12 protocol required by this target."""

        return "packed_nv12" if self.target == "x5" else "split_nv12"


@dataclass(frozen=True)
class ModelSelection:
    """A known artifact selected for one concrete target."""

    asset_id: str
    variant: str
    target: str
    model_path: Path
    contract: ClassificationContract
    sample_id: str
    explicit_model_path: bool = False


@dataclass(frozen=True)
class ModelBinding:
    """Validated connection between a selection and runtime tensor names.

    ``output_shape`` is the observed metadata shape (diagnostics); validation
    and normalization follow the H4 squeeze rule through
    :func:`score_vector_shape`/:func:`normalise_score_vector`.
    ``output_quants`` snapshots the runtime's per-output quantization
    descriptors at bind time so ``post_process`` can execute the declared
    ``output_transform`` without reaching back into the runner.
    """

    selection: ModelSelection
    contract: ClassificationContract
    model_name: str
    input_names: tuple[str, ...]
    input_shapes: Mapping[str, tuple[int, ...]]
    output_name: str
    output_shape: tuple[int, ...]
    output_dtype: str
    output_transform: str
    output_quants: Mapping[str, Any]
    y_input_name: Optional[str] = None
    uv_input_name: Optional[str] = None


def list_available_assets(target: Optional[str] = None) -> tuple[AssetRecord, ...]:
    """Return the finite pilot assets read from the existing manifests.

    ``target=None`` or ``target="auto"`` is intentionally host-independent so
    the listing command can run on a workstation.  ``s100p`` returns no rows:
    no ResNet18 asset for that target is present in the source manifest.
    """

    records = _manifest_asset_records()
    if target is None or str(target).strip().lower() == "auto":
        return records
    key = _normalise_target(target)
    return tuple(record for record in records if record.target == key)


def resolve_selection(
    target: str = "auto",
    *,
    asset_id: Optional[str] = None,
    variant: Optional[str] = None,
    model_path: Optional[str | Path] = None,
    soc_name: Optional[str] = None,
    board_type: Optional[str] = None,
) -> ModelSelection:
    """Resolve one published pilot asset and its source-proven contract.

    ``model_path`` is accepted only with an exact qualified manifest reference.
    This prevents a custom file from inheriting an input protocol because its
    basename happens to resemble ``resnet18``.
    """

    from samples._shared.platforms import resolve_target

    if model_path is not None and asset_id is None:
        raise UnsupportedAssetError(
            "An explicit model_path requires --asset-id with an exact "
            "group:sample:filename manifest reference."
        )
    try:
        resolved_target = resolve_target(
            target, soc_name=soc_name, board_type=board_type
        )
    except ValueError as exc:
        raise UnsupportedAssetError(str(exc)) from exc

    requested_variant = variant.lower() if variant is not None else None
    if requested_variant is not None and requested_variant not in SUPPORTED_VARIANTS:
        raise UnsupportedAssetError(
            f"Unknown pilot ResNet variant {variant!r}; only resnet18 is in P1."
        )

    records = list_available_assets(resolved_target)
    if asset_id is None:
        requested_variant = requested_variant or "resnet18"
        matches = [record for record in records if record.variant == requested_variant]
    else:
        # Match the exact qualified reference.  Bare model/sample IDs are not
        # silently promoted to physical artifact identities.
        matches = [record for record in records if record.asset_id == asset_id]
        if requested_variant is not None:
            matches = [record for record in matches if record.variant == requested_variant]

    if len(matches) != 1:
        available = ", ".join(record.asset_id for record in records) or "none"
        raise UnsupportedAssetError(
            f"No published pilot asset matches target={resolved_target!r}, "
            f"asset_id={asset_id!r}, variant={requested_variant!r}; "
            f"available qualified references: {available}."
        )

    record = matches[0]
    contract = _contract_for(record)
    if model_path is None:
        path = _SAMPLE_DIR / "model" / record.filename
        explicit = False
    else:
        path = Path(model_path).expanduser()
        explicit = True
    return ModelSelection(
        asset_id=record.asset_id,
        variant=record.variant,
        target=record.target,
        model_path=path,
        contract=contract,
        sample_id=record.sample_id,
        explicit_model_path=explicit,
    )


def bind_model(
    selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]
) -> ModelBinding:
    """Validate actual runtime metadata against the finite pilot contract."""

    expected_records = [
        record
        for record in _manifest_asset_records()
        if record.asset_id == selection.asset_id
        and record.variant == selection.variant
        and record.target == selection.target
    ]
    if len(expected_records) != 1:
        raise UnsupportedAssetError(
            f"Selection {selection.asset_id!r} is not published in the pilot "
            f"for {selection.target!r}."
        )

    facts = (
        metadata
        if isinstance(metadata, RuntimeMetadata)
        else RuntimeMetadata.from_mapping(metadata)
    )
    semantics = facts.output_semantics
    if isinstance(semantics, Mapping):
        semantics = semantics.get(facts.output_names[0]) if facts.output_names else None
    if semantics is not None and str(semantics).lower() not in KNOWN_OUTPUT_SEMANTICS:
        raise MetadataMismatchError(
            f"Runtime reported unsupported output semantics {semantics!r}."
        )
    if len(facts.output_names) != 1:
        raise MetadataMismatchError(
            f"Expected one classification output, found {len(facts.output_names)}."
        )
    output_name = facts.output_names[0]
    output_shape = facts.output_shapes.get(output_name)
    if not score_vector_shape(output_shape, selection.contract.class_count):
        raise MetadataMismatchError(
            f"Output {output_name!r} shape {output_shape!r} does not squeeze to "
            f"the known ({selection.contract.class_count},) score vector "
            "(H4 rank rule: singleton batch/spatial dims collapse, batch must "
            "be one)."
        )

    if output_name not in facts.output_dtypes:
        raise MetadataMismatchError(
            f"Runtime metadata is missing the dtype for output {output_name!r}."
        )
    output_dtype = canonicalise_dtype(facts.output_dtypes[output_name])
    transform = validate_output_transform(selection.contract.output_transform)
    output_quants = facts.output_quants
    if transform == "raw_f32":
        if output_dtype != "float32":
            raise MetadataMismatchError(
                f"The declared raw_f32 contract accepts only F32 output "
                f"tensors; {output_name!r} reported {output_dtype!r}. "
                "Quantized artifacts must declare the 'dequant' transform."
            )
        if output_name in output_quants:
            raise MetadataMismatchError(
                "The declared raw_f32 contract does not accept output "
                "quantization metadata."
            )
    else:  # dequant (reachable only for contracts that declare it)
        if output_name not in output_quants:
            raise MetadataMismatchError(
                f"The declared dequant contract requires a quantization "
                f"descriptor for output {output_name!r}."
            )
        if output_dtype not in {"int8", "uint8", "int16", "int32", "float32"}:
            raise MetadataMismatchError(
                f"Output {output_name!r} dtype {output_dtype!r} is not a "
                "dequantizable tensor dtype."
            )

    contract = selection.contract
    if contract_input_is_packed(contract):
        if len(facts.input_names) != 1:
            raise MetadataMismatchError(
                f"X5 packed NV12 expects one input, found {len(facts.input_names)}."
            )
        input_name = facts.input_names[0]
        shape = facts.input_shapes.get(input_name)
        expected_input = (1, 3, contract.input_height, contract.input_width)
        if tuple(shape or ()) != expected_input:
            raise MetadataMismatchError(
                f"X5 input {input_name!r} shape {shape!r} does not match the "
                f"observed NCHW {expected_input} source geometry."
            )
        _validate_input_dtype(facts, input_name, packed=True)
        return ModelBinding(
            selection=selection,
            contract=contract,
            model_name=facts.model_name,
            input_names=facts.input_names,
            input_shapes=facts.input_shapes,
            output_name=output_name,
            output_shape=tuple(output_shape or ()),
            output_dtype=output_dtype or "",
            output_transform=transform,
            output_quants=output_quants,
        )

    if contract.input_protocol == "split_nv12":
        if len(facts.input_names) != 2:
            raise MetadataMismatchError(
                f"S-series split NV12 expects two inputs, found {len(facts.input_names)}."
            )
        y_names = [
            name
            for name in facts.input_names
            if _is_y_shape(facts.input_shapes.get(name), contract.input_height, contract.input_width)
        ]
        uv_names = [
            name
            for name in facts.input_names
            if _is_uv_shape(facts.input_shapes.get(name), contract.input_height, contract.input_width)
        ]
        if len(y_names) != 1 or len(uv_names) != 1 or y_names[0] == uv_names[0]:
            raise MetadataMismatchError(
                "S-series inputs must expose one Y shape (1,H,W,1) and one "
                "UV shape (1,H/2,W/2,2); names/order are not guessed."
            )
        for name in (y_names[0], uv_names[0]):
            _validate_input_dtype(facts, name)
        return ModelBinding(
            selection=selection,
            contract=contract,
            model_name=facts.model_name,
            input_names=facts.input_names,
            input_shapes=facts.input_shapes,
            output_name=output_name,
            output_shape=tuple(output_shape or ()),
            output_dtype=output_dtype or "",
            output_transform=transform,
            output_quants=output_quants,
            y_input_name=y_names[0],
            uv_input_name=uv_names[0],
        )

    raise MetadataMismatchError(
        f"Unsupported input protocol {selection.contract.input_protocol!r}."
    )


def contract_input_is_packed(contract: ClassificationContract) -> bool:
    """Return whether the bound source uses one packed NV12 tensor."""

    return contract.input_protocol == "packed_nv12"


def _contract_for(record: AssetRecord) -> ClassificationContract:
    return ClassificationContract(
        asset_id=record.asset_id,
        variant=record.variant,
        target=record.target,
        model_format=record.model_format,
        input_protocol=record.input_protocol,
        input_height=224,
        input_width=224,
        class_count=1000,
        # Both published families return F32 score tensors; int8 outputs with
        # output_quants would require the 'dequant' transform (H1) and are a
        # declared contract change, not an implicit fallback.
        output_transform="raw_f32",
        output_semantics="unverified_score_vector",
        output_score_policy="legacy_softmax",
        resize_type=1,
        # X5 explicitly passes INTER_LINEAR for direct resize; the S wrappers
        # use their helper's INTER_NEAREST default.  Both source helpers leave
        # the letterbox cv2.resize interpolation unspecified, which means
        # OpenCV's default INTER_LINEAR behavior.
        resize_interpolation="linear" if record.target == "x5" else "nearest",
        letterbox_interpolation="linear",
        source_manifest=record.source_manifest,
    )


def score_vector_shape(shape: Sequence[int] | None, class_count: int) -> bool:
    """Apply the H4 output rank rule to one observed output shape.

    A classification artifact may spell its score vector ``(1000,)``,
    ``(1, 1000)`` (S-series) or ``(1, 1000, 1, 1)`` (X5 NCHW-style): all
    singleton dimensions collapse and the remainder must be exactly the class
    count.  A non-singleton batch or an extra real dimension is rejected —
    the rule never silently flattens ambiguous layouts.
    """

    try:
        dims = tuple(int(dimension) for dimension in (shape or ()))
    except (TypeError, ValueError):
        return False
    if any(dimension <= 0 for dimension in dims):
        return False
    squeezed = tuple(dimension for dimension in dims if dimension != 1)
    return squeezed == (int(class_count),)


def normalise_score_vector(array: Any) -> Any:
    """Squeeze one validated score tensor to its canonical 1-D form."""

    import numpy as np

    squeezed = np.squeeze(np.asarray(array))
    if squeezed.ndim != 1:
        raise MetadataMismatchError(
            f"Output does not squeeze to one score vector; got shape "
            f"{np.asarray(array).shape}."
        )
    return squeezed


def _manifest_asset_records() -> tuple[AssetRecord, ...]:
    """Read only ResNet18 rows through the shared manifest asset API."""

    try:
        from samples._shared.assets import list_assets
    except ImportError as exc:  # pragma: no cover - checkout-integrity guard
        raise ManifestAssetError(
            "The shared manifest asset resolver is required for the pilot."
        ) from exc

    records: list[AssetRecord] = []
    try:
        manifest_rows = (("x5", "resnet"), ("s", "resnet18"))
        for group, sample_id in manifest_rows:
            for asset in list_assets(group, sample_id):
                if group == "x5":
                    target = "x5"
                else:
                    pieces = asset.filename.split("/", 1)
                    if not pieces or pieces[0] not in ("s100", "s600"):
                        continue
                    target = pieces[0]
                records.append(
                    AssetRecord(
                        asset_id=asset.reference,
                        variant="resnet18",
                        target=target,
                        filename=asset.filename,
                        model_format=str(asset.format).lower(),
                        source_manifest=asset.source_path,
                        sample_id=asset.sample_id,
                    )
                )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ManifestAssetError(f"Could not read ResNet18 manifest assets: {exc}") from exc
    if not records:
        raise ManifestAssetError("No ResNet18 asset rows were found in the manifests.")
    unique: dict[str, AssetRecord] = {}
    for record in records:
        if record.asset_id in unique and unique[record.asset_id] != record:
            raise ManifestAssetError(f"Duplicate qualified asset reference {record.asset_id!r}.")
        unique[record.asset_id] = record
    return tuple(unique.values())


def _normalise_target(value: str) -> str:
    key = (value or "").strip().lower()
    if key not in SUPPORTED_TARGETS:
        raise UnsupportedAssetError(
            f"Unknown concrete target {value!r}; use x5/s100/s100p/s600 or auto."
        )
    return key


def _validate_input_dtype(
    facts: RuntimeMetadata, name: str, *, packed: bool = False
) -> None:
    if name not in facts.input_dtypes:
        raise MetadataMismatchError(
            f"Runtime metadata is missing the dtype for input {name!r}."
        )
    dtype = canonicalise_dtype(facts.input_dtypes[name])
    allowed = {"uint8", "nv12"} if packed else {"uint8"}
    if dtype is not None and dtype not in allowed:
        raise MetadataMismatchError(
            f"NV12 input {name!r} has unsupported dtype {dtype!r}; "
            f"expected one of {sorted(allowed)}."
        )


def _is_y_shape(shape: Optional[Sequence[int]], height: int, width: int) -> bool:
    return tuple(shape or ()) == (1, height, width, 1)


def _is_uv_shape(shape: Optional[Sequence[int]], height: int, width: int) -> bool:
    return tuple(shape or ()) == (1, height // 2, width // 2, 2)


__all__ = [
    "AssetRecord",
    "BindingError",
    "ClassificationContract",
    "KNOWN_OUTPUT_SEMANTICS",
    "LEGACY_SCORE_POLICIES",
    "ManifestAssetError",
    "MetadataMismatchError",
    "ModelBinding",
    "ModelSelection",
    "OUTPUT_TRANSFORMS",
    "RuntimeMetadata",
    "SUPPORTED_TARGETS",
    "SUPPORTED_VARIANTS",
    "UnsupportedAssetError",
    "bind_model",
    "contract_input_is_packed",
    "list_available_assets",
    "normalise_score_vector",
    "resolve_selection",
    "score_vector_shape",
]
