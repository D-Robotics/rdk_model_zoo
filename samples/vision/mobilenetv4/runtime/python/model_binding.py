"""Finite model and tensor contracts for the unified MobileNetV4 sample.

The publication manifests remain the authority for asset facts.  This module
declares the MobileNetV4 contract table — the variant/target facts proven during
migration — and re-exports the shared classification binding machinery
(:mod:`samples._shared.cls_binding`) under this sample's import path.  It
does not import a board SDK, inspect model bytes, or infer a protocol from a
filename supplied by a caller.

Variants ``small`` and ``medium``.  X5 source: rdk_x5 @ac11571
(``MobileNetV4_conv_{small,medium}_224x224_nv12.bin``; both 224).
S source: rdk_s @380e1a2 (small 224, medium 256 — the medium S
artifact is 256x256, unlike the 224 X5 one).

Score policy: The published artifacts declare raw logits; a stable softmax is
applied in post-processing before Top-K on both platforms.
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
#: MobileNetV4 variants published across the manifests.
SUPPORTED_VARIANTS = ('small', 'medium')
_SAMPLE_DIR = Path(__file__).resolve().parents[2]

#: Platform deployment profiles for this sample (H5).  X5 publishes flat
#: ``.bin`` artifacts; S100/S600 publish ``.hbm`` artifacts under the shared
#: ``MobileNet`` archive directory; S100P publishes none (the legacy S
#: download script silently fell back to s100, which this sample rejects).
PLATFORMS = classification_profiles(
    url_prefix_s="rdk_s100/MobileNet",
)

_SOFTMAX_224_DIRECT_LINEAR_FACTS = VariantFacts(
    input_height=224,
    input_width=224,
    output_score_policy="softmax",
    output_semantics="source_declared_logits",
    resize_type=0,  # direct resize (source default)
    resize_interpolation="linear",
)

_SOFTMAX_224_LETTERBOX_NEAREST_FACTS = VariantFacts(
    input_height=224,
    input_width=224,
    output_score_policy="softmax",
    output_semantics="source_declared_logits",
    resize_type=1,  # letterbox (source default)
    resize_interpolation="nearest",
)

_SOFTMAX_256_LETTERBOX_NEAREST_FACTS = VariantFacts(
    input_height=256,
    input_width=256,
    output_score_policy="softmax",
    output_semantics="source_declared_logits",
    resize_type=1,  # letterbox (source default)
    resize_interpolation="nearest",
)

_FACTS = {
        ('small', 'x5'): _SOFTMAX_224_DIRECT_LINEAR_FACTS,
        ('small', 's100'): _SOFTMAX_224_LETTERBOX_NEAREST_FACTS,
        ('small', 's600'): _SOFTMAX_224_LETTERBOX_NEAREST_FACTS,
        ('medium', 'x5'): _SOFTMAX_224_DIRECT_LINEAR_FACTS,
        ('medium', 's100'): _SOFTMAX_256_LETTERBOX_NEAREST_FACTS,
        ('medium', 's600'): _SOFTMAX_256_LETTERBOX_NEAREST_FACTS,
}

BINDING_TABLE = SampleBindingTable(
    sample_dir=_SAMPLE_DIR,
    manifest_rows=(
        ('x5', 'mobilenetv4'),
        ('s', 'mobilenetv4'),
    ),
    filename_variants={
        'MobileNetV4_conv_small_224x224_nv12.bin': 'small',
        'MobileNetV4_conv_medium_224x224_nv12.bin': 'medium',
        's100/mobilenetv4_small_224x224_nv12.hbm': 'small',
        's600/mobilenetv4_small_224x224_nv12.hbm': 'small',
        's100/mobilenetv4_medium_256x256_nv12.hbm': 'medium',
        's600/mobilenetv4_medium_256x256_nv12.hbm': 'medium',
    },
    default_variant='small',
    facts=_FACTS,
)


def list_available_assets(target: Optional[str] = None) -> tuple[AssetRecord, ...]:
    """Return the finite sample assets read from the existing manifests.

    ``target=None`` or ``target="auto"`` is intentionally host-independent so
    the listing command can run on a workstation.  ``s100p`` returns no rows:
    no MobileNetV4 asset for that target is present in the source manifest.
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
    """Resolve one published MobileNetV4 asset and its source-proven contract.

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
    """Validate actual runtime metadata against the MobileNetV4 contract table."""

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
