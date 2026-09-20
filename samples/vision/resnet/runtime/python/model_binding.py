"""Finite model and tensor contracts for the unified ResNet sample.

The publication manifests remain the authority for asset facts.  This module
declares the ResNet contract table — the variant/target facts proven during
migration — and re-exports the shared classification binding machinery
(:mod:`samples._shared.cls_binding`) under this sample's import path.  It
does not import a board SDK, inspect model bytes, or infer a protocol from a
filename supplied by a caller.

Variants: ``resnet18`` (X5 + S100/S600, the P1 pilot) and ``resnet50`` /
``resnet152`` (S100/S600 only; the X5 delivery never published them).  B1
migrated the S-side sources at rdk_s @380e1a2 ``samples/vision/resnet{50,152}``;
their legacy manifest IDs are kept and now point here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from samples._shared import cls_binding
from samples._shared.cls_binding import (  # noqa: F401 - re-exported surface
    OUTPUT_TRANSFORMS,
    AssetRecord,
    BindingError,
    ClassificationContract,
    KNOWN_OUTPUT_SEMANTICS,
    ManifestAssetError,
    ModelBinding,
    ModelSelection,
    RuntimeMetadata,
    SCORE_POLICIES,
    SUPPORTED_TARGETS as _SHARED_TARGETS,
    SampleBindingTable,
    UnsupportedAssetError,
    VariantFacts,
    contract_input_is_packed,
    normalise_score_vector,
    score_vector_shape,
)
from samples._shared.cls_binding import MetadataMismatchError  # noqa: F401


#: Targets the shared classification machinery can address.
SUPPORTED_TARGETS = _SHARED_TARGETS
#: ResNet variants published across the manifests.
SUPPORTED_VARIANTS = ("resnet18", "resnet50", "resnet152")
#: Kept as the pilot-era name for the ``legacy_softmax`` policy set.
LEGACY_SCORE_POLICIES = ("legacy_softmax",)
_SAMPLE_DIR = Path(__file__).resolve().parents[2]

_X5_FACTS = VariantFacts(
    input_height=224,
    input_width=224,
    class_count=1000,
    output_transform="raw_f32",
    # resnet18 keeps the pilot's evidence shape: board output looks normalized
    # already and the graph/wrapper ownership of softmax is not proven.
    output_semantics="unverified_score_vector",
    output_score_policy="legacy_softmax",
    resize_type=1,
    resize_interpolation="linear",
    letterbox_interpolation="linear",
)
_S_RESNET18_FACTS = VariantFacts(
    input_height=224,
    input_width=224,
    class_count=1000,
    output_transform="raw_f32",
    output_semantics="unverified_score_vector",
    output_score_policy="legacy_softmax",
    resize_type=1,
    resize_interpolation="nearest",
    letterbox_interpolation="linear",
)
# resnet50/resnet152 S sources document logits outputs and softmax inside
# post_process via get_topk_predictions (rdk_s @380e1a2).
_S_RESNET50_152_FACTS = VariantFacts(
    input_height=224,
    input_width=224,
    class_count=1000,
    output_transform="raw_f32",
    output_semantics="source_declared_logits",
    output_score_policy="softmax",
    resize_type=1,
    resize_interpolation="nearest",
    letterbox_interpolation="linear",
)

BINDING_TABLE = SampleBindingTable(
    sample_dir=_SAMPLE_DIR,
    manifest_rows=(
        ("x5", "resnet"),
        ("s", "resnet18"),
        ("s", "resnet50"),
        ("s", "resnet152"),
    ),
    filename_variants={
        "resnet18_224x224_nv12.bin": "resnet18",
        "s100/resnet18_224x224_nv12.hbm": "resnet18",
        "s600/resnet18_224x224_nv12.hbm": "resnet18",
        "s100/resnet50_224x224_nv12.hbm": "resnet50",
        "s600/resnet50_224x224_nv12.hbm": "resnet50",
        "s100/resnet152_224x224_nv12.hbm": "resnet152",
        "s600/resnet152_224x224_nv12.hbm": "resnet152",
    },
    default_variant="resnet18",
    facts={
        ("resnet18", "x5"): _X5_FACTS,
        ("resnet18", "s100"): _S_RESNET18_FACTS,
        ("resnet18", "s600"): _S_RESNET18_FACTS,
        ("resnet50", "s100"): _S_RESNET50_152_FACTS,
        ("resnet50", "s600"): _S_RESNET50_152_FACTS,
        ("resnet152", "s100"): _S_RESNET50_152_FACTS,
        ("resnet152", "s600"): _S_RESNET50_152_FACTS,
    },
)


def list_available_assets(target: Optional[str] = None) -> tuple[AssetRecord, ...]:
    """Return the finite sample assets read from the existing manifests.

    ``target=None`` or ``target="auto"`` is intentionally host-independent so
    the listing command can run on a workstation.  ``s100p`` returns no rows:
    no ResNet asset for that target is present in the source manifest.
    """

    return cls_binding.list_assets(BINDING_TABLE, target)


def resolve_selection(
    target: str = "auto",
    *,
    asset_id: Optional[str] = None,
    variant: Optional[str] = None,
    model_path: Optional[str | Path] = None,
    soc_name: Optional[str] = None,
    board_type: Optional[str] = None,
) -> ModelSelection:
    """Resolve one published ResNet asset and its source-proven contract.

    ``model_path`` is accepted only with an exact qualified manifest reference.
    """

    return cls_binding.resolve_selection(
        BINDING_TABLE,
        target,
        asset_id=asset_id,
        variant=variant,
        model_path=model_path,
        soc_name=soc_name,
        board_type=board_type,
    )


def bind_model(
    selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]
) -> ModelBinding:
    """Validate actual runtime metadata against the ResNet contract table."""

    return cls_binding.bind_model(BINDING_TABLE, selection, metadata)


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
    "SCORE_POLICIES",
    "SUPPORTED_TARGETS",
    "SUPPORTED_VARIANTS",
    "SampleBindingTable",
    "UnsupportedAssetError",
    "VariantFacts",
    "bind_model",
    "contract_input_is_packed",
    "list_available_assets",
    "normalise_score_vector",
    "resolve_selection",
    "score_vector_shape",
]
