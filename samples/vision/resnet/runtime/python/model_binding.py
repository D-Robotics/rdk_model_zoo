"""Finite model and tensor contracts for the ResNet classification pilot.

The publication manifests remain the authority for asset facts.  This module
only adds the small, source-proven execution contract needed by the pilot.  It
does not import a board SDK, inspect model bytes, or infer a protocol from a
filename supplied by a caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


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


class MetadataMismatchError(BindingError):
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
    """

    asset_id: str
    variant: str
    target: str
    model_format: str
    input_protocol: str
    input_height: int
    input_width: int
    class_count: int
    output_shape: tuple[int, ...]
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
class RuntimeMetadata:
    """Runtime facts used to validate one loaded model instance."""

    model_name: str
    input_names: tuple[str, ...]
    input_shapes: Mapping[str, tuple[int, ...]]
    output_names: tuple[str, ...]
    output_shapes: Mapping[str, tuple[int, ...]]
    input_dtypes: Mapping[str, str]
    output_dtypes: Mapping[str, str]
    output_semantics: Optional[str] = None
    output_scales: Mapping[str, float] = field(default_factory=dict)
    output_zero_points: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RuntimeMetadata":
        """Create metadata from a flat or one-model nested mapping."""

        model_name = str(values.get("model_name") or values.get("name") or "model")

        def field(name: str, default: Any) -> Any:
            value = values.get(name, default)
            if isinstance(value, Mapping) and model_name in value:
                return value[model_name]
            return value

        input_names = tuple(str(value) for value in field("input_names", ()))
        output_names = tuple(str(value) for value in field("output_names", ()))
        input_shapes = _normalise_shapes(field("input_shapes", {}))
        output_shapes = _normalise_shapes(field("output_shapes", {}))
        input_dtypes = _normalise_dtypes(field("input_dtypes", {}))
        output_dtypes = _normalise_dtypes(field("output_dtypes", {}))
        output_scales = _normalise_numbers(field("output_scales", {}))
        output_zero_points = _normalise_numbers(field("output_zero_points", {}))
        semantics = field("output_semantics", None)
        if isinstance(semantics, Mapping):
            semantics = semantics.get(output_names[0]) if output_names else None
        return cls(
            model_name=model_name,
            input_names=input_names,
            input_shapes=input_shapes,
            output_names=output_names,
            output_shapes=output_shapes,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
            output_semantics=str(semantics).lower() if semantics is not None else None,
            output_scales=output_scales,
            output_zero_points=output_zero_points,
        )

    @classmethod
    def from_runtime(cls, runtime: Any) -> "RuntimeMetadata":
        """Read public metadata attributes exposed by ``hbm_runtime``."""

        names = getattr(runtime, "model_names", None)
        if not names:
            raise MetadataMismatchError("Runtime did not expose model_names.")
        model_name = str(names[0])

        def model_field(name: str, default: Any) -> Any:
            value = getattr(runtime, name, default)
            if isinstance(value, Mapping) and model_name in value:
                return value[model_name]
            return value

        return cls.from_mapping(
            {
                "model_name": model_name,
                "input_names": model_field("input_names", ()),
                "input_shapes": model_field("input_shapes", {}),
                "output_names": model_field("output_names", ()),
                "output_shapes": model_field("output_shapes", {}),
                "input_dtypes": model_field("input_dtypes", {}),
                "output_dtypes": model_field("output_dtypes", {}),
                "output_semantics": model_field("output_semantics", None),
                # These fields are optional.  The pilot accepts F32 outputs
                # only, so it never guesses a quantization scale.
                "output_scales": model_field("output_scales", {}),
                "output_zero_points": model_field("output_zero_points", {}),
            }
        )


@dataclass(frozen=True)
class ModelBinding:
    """Validated connection between a selection and runtime tensor names."""

    selection: ModelSelection
    contract: ClassificationContract
    model_name: str
    input_names: tuple[str, ...]
    input_shapes: Mapping[str, tuple[int, ...]]
    output_name: str
    output_shape: tuple[int, ...]
    output_dtype: str
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
    if facts.output_semantics is not None and facts.output_semantics not in KNOWN_OUTPUT_SEMANTICS:
        raise MetadataMismatchError(
            f"Runtime reported unsupported output semantics {facts.output_semantics!r}."
        )
    if len(facts.output_names) != 1:
        raise MetadataMismatchError(
            f"Expected one classification output, found {len(facts.output_names)}."
        )
    output_name = facts.output_names[0]
    output_shape = facts.output_shapes.get(output_name)
    if tuple(output_shape or ()) != selection.contract.output_shape:
        raise MetadataMismatchError(
            f"Output {output_name!r} shape {output_shape!r} does not match the "
            f"known {selection.contract.output_shape} score-vector shape."
        )

    if output_name not in facts.output_dtypes:
        raise MetadataMismatchError(
            f"Runtime metadata is missing the dtype for output {output_name!r}."
        )
    output_dtype = _canonical_dtype(facts.output_dtypes[output_name])
    if output_dtype != "float32":
        raise MetadataMismatchError(
            f"The P1 pilot accepts only F32 output tensors; {output_name!r} "
            f"reported {output_dtype!r}. No quantization scale is guessed."
        )
    if facts.output_scales.get(output_name) is not None or facts.output_zero_points.get(output_name) is not None:
        raise MetadataMismatchError(
            "The P1 F32 contract does not accept output quantization metadata."
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
            output_shape=tuple(output_shape),
            output_dtype=output_dtype,
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
            output_shape=tuple(output_shape),
            output_dtype=output_dtype,
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
        output_shape=(1, 1000, 1, 1) if record.target == "x5" else (1, 1000),
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


def _normalise_shapes(values: Any) -> dict[str, tuple[int, ...]]:
    if not isinstance(values, Mapping):
        return {}
    result: dict[str, tuple[int, ...]] = {}
    for name, shape in values.items():
        try:
            result[str(name)] = tuple(int(dimension) for dimension in shape)
        except (TypeError, ValueError):
            result[str(name)] = ()
    return result


def _normalise_dtypes(values: Any) -> dict[str, str]:
    if not isinstance(values, Mapping):
        return {}
    result: dict[str, str] = {}
    for name, dtype in values.items():
        raw = str(getattr(dtype, "name", dtype)).lower()
        if raw in {"f32", "float", "float32", "hbdnndatatype.f32"} or raw.endswith(".f32"):
            raw = "float32"
        elif raw in {"u8", "uint8", "hbdnndatatype.u8"} or raw.endswith(".u8"):
            raw = "uint8"
        elif raw in {"nv12", "hbdnndatatype.nv12"} or raw.endswith(".nv12"):
            raw = "nv12"
        result[str(name)] = raw
    return result


def _normalise_numbers(values: Any) -> dict[str, float]:
    if not isinstance(values, Mapping):
        return {}
    result: dict[str, float] = {}
    for name, value in values.items():
        try:
            result[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _validate_input_dtype(
    facts: RuntimeMetadata, name: str, *, packed: bool = False
) -> None:
    if name not in facts.input_dtypes:
        raise MetadataMismatchError(
            f"Runtime metadata is missing the dtype for input {name!r}."
        )
    dtype = _canonical_dtype(facts.input_dtypes[name])
    allowed = {"uint8", "nv12"} if packed else {"uint8"}
    if dtype is not None and dtype not in allowed:
        raise MetadataMismatchError(
            f"NV12 input {name!r} has unsupported dtype {dtype!r}; "
            f"expected one of {sorted(allowed)}."
        )


def _canonical_dtype(dtype: Any) -> str:
    """Normalize a direct dataclass value like the mapping path does."""

    return next(iter(_normalise_dtypes({"dtype": dtype}).values()))


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
    "RuntimeMetadata",
    "SUPPORTED_TARGETS",
    "SUPPORTED_VARIANTS",
    "UnsupportedAssetError",
    "bind_model",
    "contract_input_is_packed",
    "list_available_assets",
    "resolve_selection",
]
