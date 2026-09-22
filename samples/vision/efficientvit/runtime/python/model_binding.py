"""Finite model and tensor contracts for the unified EfficientViT sample.

The publication manifests remain the authority for asset facts.  This module
declares the EfficientViT contract table — the variant/target facts proven
during migration — and re-exports the shared classification binding machinery
(:mod:`samples._shared.cls_binding`) under this sample's import path.  It
does not import a board SDK, inspect model bytes, or infer a protocol from a
filename supplied by a caller.

X5 source: rdk_x5 @ac11571 (single variant m5 — the MSRA EfficientViT
"Cascaded Group Attention" series — 224x224, letterbox with linear
interpolation).  The S branch publishes no EfficientViT asset, so every S
target resolves to zero published assets — selection is an explicit error,
never a cross-platform fallback.  Neither source ships a C++ runtime for
this sample.

Score policy: the source wrapper treats the graph output as ImageNet-1k
logits and applies ``scipy.special.softmax`` before Top-K, so the contract
semantics is ``source_declared_logits`` with the ``softmax`` score policy.
The source entrypoint's default model is the m5 artifact, which the table's
default variant preserves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

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
from samples._shared.platform_profile import (
    PlatformProfile,
    UnsupportedProfileError,
    classification_profiles,
    resolve_profile,
)


#: Targets the shared classification machinery can address.
SUPPORTED_TARGETS = _SHARED_TARGETS
#: EfficientViT variants published across the manifests (X5 m5 only).
SUPPORTED_VARIANTS = ('m5',)
_SAMPLE_DIR = Path(__file__).resolve().parents[2]

#: Platform deployment profiles for this sample (H5).  X5 publishes flat
#: ``.bin`` artifacts; the S manifests carry no EfficientViT row, so S
#: selection errors with zero published assets.  ``url_prefix_s`` is
#: structurally required by the shared four-profile builder but is never
#: consulted for this sample: no S manifest row exists, so no S URL can ever
#: resolve.  No C++ runtime exists on any target for this sample.
PLATFORMS = classification_profiles(url_prefix_s="rdk_s100/EfficientViT")


def _x5_224_facts() -> VariantFacts:
    return VariantFacts(
        input_height=224,
        input_width=224,
        output_semantics="source_declared_logits",
        output_score_policy="softmax",
        resize_type=1,  # letterbox (source default)
        resize_interpolation="linear",
        letterbox_interpolation="linear",
    )


#: The single published variant is 224x224 (manifest filename and YAML
#: input agree).
_FACTS = {
    ('m5', 'x5'): _x5_224_facts(),
}

BINDING_TABLE = SampleBindingTable(
    sample_dir=_SAMPLE_DIR,
    manifest_rows=(
        ('x5', 'efficientvit'),
    ),
    filename_variants={
        'EfficientViT_m5_224x224_nv12.bin': 'm5',
    },
    default_variant='m5',
    facts=_FACTS,
)


def list_available_assets(target: Optional[str] = None) -> tuple[AssetRecord, ...]:
    """Return the finite sample assets read from the existing manifests.

    ``target=None`` or ``target="auto"`` is intentionally host-independent so
    the listing command can run on a workstation.  Every S target (S100,
    S100P, S600) returns no rows: no EfficientViT asset for those targets
    is present in the source manifest.
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
    """Resolve one published EfficientViT asset and its source-proven contract.

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
    """Validate actual runtime metadata against the EfficientViT contract table."""

    return cls_binding.bind_model(BINDING_TABLE, selection, metadata)


__all__ = [
    "AssetRecord",
    "BINDING_TABLE",
    "BindingError",
    "ClassificationContract",
    "KNOWN_OUTPUT_SEMANTICS",
    "ManifestAssetError",
    "MetadataMismatchError",
    "ModelBinding",
    "ModelSelection",
    "OUTPUT_TRANSFORMS",
    "PLATFORMS",
    "PlatformProfile",
    "RuntimeMetadata",
    "SCORE_POLICIES",
    "SUPPORTED_TARGETS",
    "SUPPORTED_VARIANTS",
    "SampleBindingTable",
    "UnsupportedAssetError",
    "UnsupportedProfileError",
    "VariantFacts",
    "bind_model",
    "classification_profiles",
    "contract_input_is_packed",
    "list_available_assets",
    "normalise_score_vector",
    "resolve_profile",
    "resolve_selection",
    "score_vector_shape",
]
