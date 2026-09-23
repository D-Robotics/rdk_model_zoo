# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VargConvNet artifact contracts from rdk_x5 @ac115717197920355fc390bb04299b20e6436864.

The 1 published variants declare 224x224 ImageNet-1k classification.
The source applies scipy softmax to F32 scores after linear letterbox and
packed NV12 preprocessing. These are source contracts, not observed board
metadata; the lazy runner validates the actual metadata before inference.
There is no published S-series asset or C++ runtime for this sample.
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
#: Published variants; filenames preserve release case.
SUPPORTED_VARIANTS = ('vargconvnet',)
_SAMPLE_DIR = Path(__file__).resolve().parents[2]
# Identity profiles do not create assets for the unpublished S targets.
PLATFORMS = classification_profiles(url_prefix_s="rdk_s100/VargConvNet")
BINDING_TABLE = SampleBindingTable(
    sample_dir=_SAMPLE_DIR,
    manifest_rows=(("x5", "vargconvnet"),),
    filename_variants={
        'vargconvnet_224x224_nv12.bin': 'vargconvnet',
    },
    default_variant='vargconvnet',
    facts={(v,"x5"): VariantFacts(
        input_height=224, input_width=224,
        output_semantics="source_declared_logits", output_score_policy="softmax",
        resize_type=1, resize_interpolation="linear", letterbox_interpolation="linear",
    ) for v in SUPPORTED_VARIANTS},
)


def list_available_assets(target: Optional[str] = None) -> tuple[AssetRecord, ...]:
    """Return the finite sample assets read from the existing manifests.

    ``target=None`` or ``target="auto"`` is intentionally host-independent so
    the listing command can run on a workstation.  Every S target (S100,
    S100P, S600) returns no rows: no VargConvNet asset for those targets
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
    """Resolve one published VargConvNet asset and its source-proven contract.

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
    """Validate actual runtime metadata against the VargConvNet contract table."""

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
