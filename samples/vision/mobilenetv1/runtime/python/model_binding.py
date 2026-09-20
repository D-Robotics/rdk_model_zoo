"""Finite model and tensor contracts for the unified MobileNetV1 sample.

The publication manifests remain the authority for asset facts.  This module
declares the MobileNetV1 contract table — the variant/target facts proven during
migration — and re-exports the shared classification binding machinery
(:mod:`samples._shared.cls_binding`) under this sample's import path.  It
does not import a board SDK, inspect model bytes, or infer a protocol from a
filename supplied by a caller.

X5 source: rdk_x5 @ac11571 ``samples/vision/mobilenetv1`` (scipy not
needed; post-softmax probabilities, no extra softmax).  S source:
rdk_s @380e1a2 ``samples/vision/mobilenetv1``.

Score policy: The published artifacts declare post-softmax probabilities at the
graph output (both platforms), so no additional softmax is applied
and the raw values are preserved.
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
#: MobileNetV1 variants published across the manifests.
SUPPORTED_VARIANTS = ('mobilenetv1',)
_SAMPLE_DIR = Path(__file__).resolve().parents[2]

#: Platform deployment profiles for this sample (H5).  X5 publishes flat
#: ``.bin`` artifacts; S100/S600 publish ``.hbm`` artifacts under the shared
#: ``MobileNet`` archive directory; S100P publishes none (the legacy S
#: download script silently fell back to s100, which this sample rejects).
PLATFORMS = classification_profiles(
    url_prefix_s="rdk_s100/MobileNet",
)

_NONE_224_DIRECT_LINEAR_FACTS = VariantFacts(
    input_height=224,
    input_width=224,
    output_score_policy="none",
    output_semantics="source_declared_probabilities",
    resize_type=0,  # direct resize (source default)
    resize_interpolation="linear",
)

_NONE_224_LETTERBOX_NEAREST_FACTS = VariantFacts(
    input_height=224,
    input_width=224,
    output_score_policy="none",
    output_semantics="source_declared_probabilities",
    resize_type=1,  # letterbox (source default)
    resize_interpolation="nearest",
)

_FACTS = {
        ('mobilenetv1', 'x5'): _NONE_224_DIRECT_LINEAR_FACTS,
        ('mobilenetv1', 's100'): _NONE_224_LETTERBOX_NEAREST_FACTS,
        ('mobilenetv1', 's600'): _NONE_224_LETTERBOX_NEAREST_FACTS,
}

BINDING_TABLE = SampleBindingTable(
    sample_dir=_SAMPLE_DIR,
    manifest_rows=(
        ('x5', 'mobilenetv1'),
        ('s', 'mobilenetv1'),
    ),
    filename_variants={
        'mobilenetv1_224x224_nv12.bin': 'mobilenetv1',
        's100/mobilenetv1_224x224_nv12.hbm': 'mobilenetv1',
        's600/mobilenetv1_224x224_nv12.hbm': 'mobilenetv1',
    },
    default_variant='mobilenetv1',
    facts=_FACTS,
)


def list_available_assets(target: Optional[str] = None) -> tuple[AssetRecord, ...]:
    """Return the finite sample assets read from the existing manifests.

    ``target=None`` or ``target="auto"`` is intentionally host-independent so
    the listing command can run on a workstation.  ``s100p`` returns no rows:
    no MobileNetV1 asset for that target is present in the source manifest.
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
    """Resolve one published MobileNetV1 asset and its source-proven contract.

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
    """Validate actual runtime metadata against the MobileNetV1 contract table."""

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
