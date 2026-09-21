"""Finite model and tensor contracts for the unified EfficientNet sample.

The publication manifests remain the authority for asset facts.  This module
declares the EfficientNet contract table — the variant/target facts proven
during migration — and re-exports the shared classification binding machinery
(:mod:`samples._shared.cls_binding`) under this sample's import path.  It
does not import a board SDK, inspect model bytes, or infer a protocol from a
filename supplied by a caller.

X5 source: rdk_x5 @ac11571 (variants B2/B3/B4, all 224x224, letterbox with
linear interpolation).  S source: rdk_s @380e1a2 (variants lite0..lite4 with
**per-variant geometry** 224/240/260/300/380, letterbox with nearest
interpolation; the legacy wrapper silently fell back to the lite0 S100 model
for every non-S600 SoC, which this sample rejects).  Neither source ships a
C++ runtime for this sample.

Score policy: both wrappers treat the graph output as ImageNet-1k logits
declared by the source (X5 wrapper docstring "single logits output"; S
``get_topk_predictions`` applies its own softmax) and apply a softmax (X5:
``scipy.special.softmax``; S: the numerically stable softmax inside
``visualize.get_topk_predictions``), so the contract semantics is
``source_declared_logits`` with the ``softmax`` score policy.
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
#: EfficientNet variants published across the manifests (X5 B-series, S lite-series).
SUPPORTED_VARIANTS = ('b2', 'b3', 'b4', 'lite0', 'lite1', 'lite2', 'lite3', 'lite4')
_SAMPLE_DIR = Path(__file__).resolve().parents[2]

#: Platform deployment profiles for this sample (H5).  X5 publishes flat
#: ``.bin`` artifacts; S100/S600 publish ``.hbm`` artifacts under the shared
#: ``EfficientNet`` archive directory; S100P publishes none (the legacy S
#: download script silently fell back to the S100 build, which this sample
#: rejects).  No C++ runtime exists on any target for this sample.
PLATFORMS = classification_profiles(url_prefix_s="rdk_s100/EfficientNet")


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


def _s_facts(size: int) -> VariantFacts:
    return VariantFacts(
        input_height=size,
        input_width=size,
        output_semantics="source_declared_logits",
        output_score_policy="softmax",
        resize_type=1,  # letterbox (source default)
        resize_interpolation="nearest",
        letterbox_interpolation="linear",
    )


#: Per-variant S geometry (manifest filenames and lite*_config.yaml prefixes
#: agree on 224/240/260/300/380); X5 variants are all 224.
_FACTS = {
    ('b2', 'x5'): _x5_224_facts(),
    ('b3', 'x5'): _x5_224_facts(),
    ('b4', 'x5'): _x5_224_facts(),
    ('lite0', 's100'): _s_facts(224),
    ('lite0', 's600'): _s_facts(224),
    ('lite1', 's100'): _s_facts(240),
    ('lite1', 's600'): _s_facts(240),
    ('lite2', 's100'): _s_facts(260),
    ('lite2', 's600'): _s_facts(260),
    ('lite3', 's100'): _s_facts(300),
    ('lite3', 's600'): _s_facts(300),
    ('lite4', 's100'): _s_facts(380),
    ('lite4', 's600'): _s_facts(380),
}

BINDING_TABLE = SampleBindingTable(
    sample_dir=_SAMPLE_DIR,
    manifest_rows=(
        ('x5', 'efficientnet'),
        ('s', 'efficientnet'),
    ),
    filename_variants={
        'EfficientNet_B2_224x224_nv12.bin': 'b2',
        'EfficientNet_B3_224x224_nv12.bin': 'b3',
        'EfficientNet_B4_224x224_nv12.bin': 'b4',
        's100/efficientnet_lite0_224x224_nv12.hbm': 'lite0',
        's100/efficientnet_lite1_240x240_nv12.hbm': 'lite1',
        's100/efficientnet_lite2_260x260_nv12.hbm': 'lite2',
        's100/efficientnet_lite3_300x300_nv12.hbm': 'lite3',
        's100/efficientnet_lite4_380x380_nv12.hbm': 'lite4',
        's600/efficientnet_lite0_224x224_nv12.hbm': 'lite0',
        's600/efficientnet_lite1_240x240_nv12.hbm': 'lite1',
        's600/efficientnet_lite2_260x260_nv12.hbm': 'lite2',
        's600/efficientnet_lite3_300x300_nv12.hbm': 'lite3',
        's600/efficientnet_lite4_380x380_nv12.hbm': 'lite4',
    },
    default_variant='b2',
    facts=_FACTS,
)


def list_available_assets(target: Optional[str] = None) -> tuple[AssetRecord, ...]:
    """Return the finite sample assets read from the existing manifests.

    ``target=None`` or ``target="auto"`` is intentionally host-independent so
    the listing command can run on a workstation.  ``s100p`` returns no rows:
    no EfficientNet asset for that target is present in the source manifest.
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
    """Resolve one published EfficientNet asset and its source-proven contract.

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
    """Validate actual runtime metadata against the EfficientNet contract table."""

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
