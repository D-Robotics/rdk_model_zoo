"""Shared binding machinery for the unified classification samples.

The publication manifests remain the authority for asset facts.  This module
provides the machinery every standardized classification sample shares:
manifest-row reading, selection resolution, and runtime-metadata validation
against a per-sample contract table.  Each sample's ``model_binding.py`` owns
its table — the variant/target facts that genuinely differ per sample — and
re-exports the machinery under its own module path.

The table is the single place where platform differences live for a sample
(the migration template's ``model_binding.py(平台差异)``).  Nothing here
imports a board SDK, inspects model bytes, or infers a protocol from a caller
supplied filename.
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

#: Targets the standardized classification delivery can address.  A concrete
#: sample only executes rows that its platform manifest actually publishes.
SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")

#: Score policies a classification contract may declare.  ``legacy_softmax``
#: and ``softmax`` are the same stable math with different epistemic status:
#: the legacy name keeps the pilot warning that X5 board evidence did not
#: prove graph semantics; ``softmax`` marks source-documented logits.  ``none``
#: preserves sources that print the raw (already-activated) values.
SCORE_POLICIES = ("legacy_softmax", "softmax", "none")

#: Runtime metadata, when a future SDK exposes this optional field, may only
#: report terms observed in the source/evidence.  An absent field remains the
#: normal path; arbitrary or "unknown" declarations are rejected.
KNOWN_OUTPUT_SEMANTICS = frozenset({"logits", "probabilities"})


class BindingError(ValueError):
    """Base class for selection and runtime contract failures."""


class UnsupportedAssetError(BindingError):
    """The requested asset/target combination is not in the sample set."""


class MetadataMismatchError(BindingError, _SharedMetadataMismatchError):
    """Runtime metadata cannot satisfy the declared sample contract."""


class ManifestAssetError(BindingError):
    """The shared publication manifest could not provide the sample asset."""


@dataclass(frozen=True)
class VariantFacts:
    """Per-(variant, target) contract facts for one classification artifact.

    These are source-proven values recorded in the sample's old→new mapping
    table during migration; defaults encode the shared classification shape
    (ImageNet-1k, F32 output, letterbox).  Geometry varies (mobilenetv4
    medium is 256×256 on S), so every published (variant, target) row is
    explicit in the sample table rather than derived.
    """

    input_height: int
    input_width: int
    class_count: int = 1000
    output_transform: str = "raw_f32"
    output_semantics: str = "unverified_score_vector"
    output_score_policy: str = "legacy_softmax"
    resize_type: int = 1
    resize_interpolation: str = "linear"
    letterbox_interpolation: str = "linear"


@dataclass(frozen=True)
class SampleBindingTable:
    """One sample's declarative manifest and contract mapping.

    Attributes:
        sample_dir: The sample root (``.../samples/<domain>/<sample>``); the
            ``model/`` directory resolves relative to it.
        manifest_rows: ``(group, sample_id)`` pairs to read, e.g.
            ``(("x5", "mobilenetv2"), ("s", "mobilenetv2"))``.
        filename_variants: Exact manifest filename → variant.  Filenames are
            the published identity (including the ``s100/``/``s600/`` prefix
            and historical letter casing); variants are never guessed from a
            basename.
        default_variant: Variant selected when neither ``--variant`` nor
            ``--asset-id`` names one.  A plain string applies to every target;
            a ``{target: variant}`` mapping selects one default per target —
            used when the source entrypoints defaulted to different models per
            platform (e.g. EfficientNet: X5 B2, S lite0).  A target absent
            from the mapping has no default, so an omitted variant fails with
            the standard "no published asset" error instead of silently
            selecting another platform's model.
        facts: ``(variant, target)`` → :class:`VariantFacts`.  Every
            published (variant, target) combination must have a row.
        s_filename_targets: Filename prefixes under the ``s`` group that map
            onto concrete targets (``s100/``, ``s600/``); other prefixes are
            skipped as unpublished for this sample.
    """

    sample_dir: Path
    manifest_rows: tuple[tuple[str, str], ...]
    filename_variants: Mapping[str, str]
    default_variant: str | Mapping[str, str]
    facts: Mapping[tuple[str, str], VariantFacts]
    s_filename_targets: tuple[str, ...] = ("s100", "s600")

    def default_variant_for(self, target: str) -> Optional[str]:
        """Return the default variant for ``target``, or ``None`` if unset.

        A mapping without the target yields ``None`` so the caller falls
        through to the standard no-published-asset error rather than picking
        another target's default.
        """

        if isinstance(self.default_variant, str):
            return self.default_variant
        return self.default_variant.get(target)


@dataclass(frozen=True)
class ClassificationContract:
    """Known input/output facts for one published classification artifact.

    ``output_semantics`` records what the migration could prove about the
    output values: ``unverified_score_vector`` (pilot evidence shape),
    ``source_declared_logits`` / ``source_declared_probabilities`` (source
    docstrings), or another sample-recorded term.  ``output_score_policy``
    selects the post-processing math (see :data:`SCORE_POLICIES`).

    ``output_transform`` (Phase 1.5 H1) declares how raw runtime outputs
    become float32 values.  Output shapes follow the H4 rank rule
    (:func:`score_vector_shape`): ``(1, 1000, 1, 1)`` and ``(1, 1000)`` are
    two spellings of one squeezable score vector, not two contracts.
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

    ``asset_id`` is the qualified reference ``group:sample:filename``.  It
    is a UI and selection reference, not a newly invented standalone catalog
    ID.
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


def list_assets(
    table: SampleBindingTable, target: Optional[str] = None
) -> tuple[AssetRecord, ...]:
    """Return the sample's assets read from the existing manifests.

    ``target=None`` or ``target="auto"`` is intentionally host-independent so
    the listing command can run on a workstation.  A target without published
    rows (e.g. ``s100p``) returns an empty tuple.
    """

    records = read_manifest_asset_records(table)
    if target is None or str(target).strip().lower() == "auto":
        return records
    key = normalise_target(target)
    return tuple(record for record in records if record.target == key)


def resolve_selection(
    table: SampleBindingTable,
    target: str = "auto",
    *,
    asset_id: Optional[str] = None,
    variant: Optional[str] = None,
    model_path: Optional[str | Path] = None,
    soc_name: Optional[str] = None,
    board_type: Optional[str] = None,
) -> ModelSelection:
    """Resolve one published sample asset and its source-proven contract.

    ``model_path`` is accepted only with an exact qualified manifest reference.
    This prevents a custom file from inheriting an input protocol because its
    basename happens to resemble a published artifact.
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

    variants = sorted(set(table.filename_variants.values()))
    requested_variant = variant.lower() if variant is not None else None
    if requested_variant is not None and requested_variant not in variants:
        raise UnsupportedAssetError(
            f"Unknown sample variant {variant!r}; known variants: "
            f"{', '.join(variants)}."
        )

    records = list_assets(table, resolved_target)
    if asset_id is None:
        if requested_variant is None:
            requested_variant = table.default_variant_for(resolved_target)
        if requested_variant is not None:
            matches = [
                record for record in records if record.variant == requested_variant
            ]
        else:
            # No default variant declared for this target: an omitted variant
            # is an explicit error even when the target happens to publish a
            # single asset — a lone candidate is not an implicit default.
            matches = []
    else:
        # Match the exact qualified reference.  Bare model/sample IDs are not
        # silently promoted to physical artifact identities.
        matches = [record for record in records if record.asset_id == asset_id]
        if requested_variant is not None:
            matches = [record for record in matches if record.variant == requested_variant]

    if len(matches) != 1:
        available = ", ".join(record.asset_id for record in records) or "none"
        raise UnsupportedAssetError(
            f"No published sample asset matches target={resolved_target!r}, "
            f"asset_id={asset_id!r}, variant={requested_variant!r}; "
            f"available qualified references: {available}."
        )

    record = matches[0]
    contract = contract_for(table, record)
    if model_path is None:
        path = table.sample_dir / "model" / record.filename
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
    table: SampleBindingTable,
    selection: ModelSelection,
    metadata: RuntimeMetadata | Mapping[str, Any],
) -> ModelBinding:
    """Validate actual runtime metadata against the sample contract table."""

    expected_records = [
        record
        for record in read_manifest_asset_records(table)
        if record.asset_id == selection.asset_id
        and record.variant == selection.variant
        and record.target == selection.target
    ]
    if len(expected_records) != 1:
        raise UnsupportedAssetError(
            f"Selection {selection.asset_id!r} is not published for "
            f"{selection.target!r}."
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
        # A vestigial quant descriptor alongside an F32 output is accepted:
        # the runtime values are already final floats and post-processing
        # consumes them directly, so the snapshot keeps the descriptor for
        # the record but never applies it (matches legacy consumers).
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
    if contract.input_protocol == "packed_nv12":
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


def contract_for(table: SampleBindingTable, record: AssetRecord) -> ClassificationContract:
    """Build the contract for one record from the sample's facts table."""

    facts = table.facts.get((record.variant, record.target))
    if facts is None:
        raise ManifestAssetError(
            f"No contract facts are recorded for variant {record.variant!r} on "
            f"{record.target!r}; the migration table is incomplete."
        )
    if facts.output_score_policy not in SCORE_POLICIES:
        raise ManifestAssetError(
            f"Score policy {facts.output_score_policy!r} is not one of "
            f"{SCORE_POLICIES}."
        )
    return ClassificationContract(
        asset_id=record.asset_id,
        variant=record.variant,
        target=record.target,
        model_format=record.model_format,
        input_protocol=record.input_protocol,
        input_height=facts.input_height,
        input_width=facts.input_width,
        class_count=facts.class_count,
        output_transform=validate_output_transform(facts.output_transform),
        output_semantics=facts.output_semantics,
        output_score_policy=facts.output_score_policy,
        resize_type=facts.resize_type,
        resize_interpolation=facts.resize_interpolation,
        letterbox_interpolation=facts.letterbox_interpolation,
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


def read_manifest_asset_records(table: SampleBindingTable) -> tuple[AssetRecord, ...]:
    """Read the sample's rows through the shared manifest asset API."""

    try:
        from samples._shared.assets import list_assets as _list_assets
    except ImportError as exc:  # pragma: no cover - checkout-integrity guard
        raise ManifestAssetError(
            "The shared manifest asset resolver is required for the sample."
        ) from exc

    records: list[AssetRecord] = []
    try:
        for group, sample_id in table.manifest_rows:
            for asset in _list_assets(group, sample_id):
                variant = table.filename_variants.get(asset.filename)
                if variant is None:
                    # Not an error: manifests may carry artifacts this sample
                    # version does not bind (they stay listed for download).
                    continue
                if group == "x5":
                    target = "x5"
                else:
                    pieces = asset.filename.split("/", 1)
                    if not pieces or pieces[0] not in table.s_filename_targets:
                        continue
                    target = pieces[0]
                records.append(
                    AssetRecord(
                        asset_id=asset.reference,
                        variant=variant,
                        target=target,
                        filename=asset.filename,
                        model_format=str(asset.format).lower(),
                        source_manifest=asset.source_path,
                        sample_id=asset.sample_id,
                    )
                )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ManifestAssetError(f"Could not read sample manifest assets: {exc}") from exc
    if not records:
        raise ManifestAssetError("No sample asset rows were found in the manifests.")
    unique: dict[str, AssetRecord] = {}
    for record in records:
        if record.asset_id in unique and unique[record.asset_id] != record:
            raise ManifestAssetError(f"Duplicate qualified asset reference {record.asset_id!r}.")
        unique[record.asset_id] = record
    return tuple(unique.values())


def normalise_target(value: str) -> str:
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
    "ManifestAssetError",
    "MetadataMismatchError",
    "ModelBinding",
    "ModelSelection",
    "OUTPUT_TRANSFORMS",
    "RuntimeMetadata",
    "SCORE_POLICIES",
    "SUPPORTED_TARGETS",
    "SampleBindingTable",
    "UnsupportedAssetError",
    "VariantFacts",
    "bind_model",
    "contract_for",
    "contract_input_is_packed",
    "list_assets",
    "normalise_score_vector",
    "normalise_target",
    "read_manifest_asset_records",
    "resolve_selection",
    "score_vector_shape",
]
