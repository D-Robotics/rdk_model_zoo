# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Finite SigLIP publication and packed-submodel contracts.

The fixed S source explicitly publishes the same eight artifacts for Nash-E
and Nash-M; the `s100/` storage directory is not a hardware fallback. Historical
evaluator tables supply size/feature geometry; actual metadata is validated
before inference. Output values keep the runtime's native numeric dtype and
shape: the source neither dequantizes nor activates SigLIP outputs.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Any
from samples._shared.assets import Asset, list_assets
from samples._shared.platforms import resolve_target
from samples._shared.runtime_meta import RuntimeMetadata, MetadataMismatchError

SUBMODELS = ('pooler_output', 'last_hidden_state')
SUPPORTED_TARGETS = ('s100', 's100p')
DEFAULT_VARIANT = 'base-patch16-224'
SAMPLE_DIR = Path(__file__).resolve().parents[2]
# size, embedding dimension, patch tokens: literal source evaluator facts.
VARIANTS = {
    'base-patch16-224': (224, 768, 196),
    'base-patch16-384': (384, 768, 576),
    'base-patch16-512': (512, 768, 1024),
    'large-patch16-256': (256, 1024, 256),
    'large-patch16-384': (384, 1024, 576),
    'so400m-patch14-224': (224, 1152, 256),
    'so400m-patch14-384': (384, 1152, 729),
    'so400m-patch16-256-i18n': (256, 1152, 256),
}
NUMERIC_DTYPES = ('float16', 'float32', 'float64', 'int8', 'uint8', 'int16', 'int32', 'int64')


@dataclass(frozen=True)
class ModelSelection:
    """Exact asset, actual execution target and selected packed submodel."""
    asset: Asset
    target: str
    variant: str
    model_path: Path
    submodel: str
    image_size: int


@dataclass(frozen=True)
class ModelBinding:
    """Validated physical shapes/dtype for one selected submodel; no decoding."""
    selection: ModelSelection
    model_name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_dtype: str


def list_available_assets(target: str | None = None) -> tuple[Asset, ...]:
    """Read eight unique manifest assets; both supported targets use these files."""
    if target not in (None, 'auto', *SUPPORTED_TARGETS):
        if target not in ('x5', 's600'):
            raise ValueError(f'Unknown target: {target}')
        return ()
    assets = list_assets('s', 'siglip')
    expected = {f's100/bpu-siglip-{v}.hbm' for v in VARIANTS}
    if {a.filename for a in assets} != expected or any(a.format != 'hbm' for a in assets):
        raise ValueError('SigLIP publication changed; review its finite contracts first.')
    return assets


def resolve_selection(target='auto', *, variant=None, asset_id=None, model_path=None,
                      submodel='pooler_output', image_size=None, soc_name=None,
                      board_type=None) -> ModelSelection:
    """Resolve source-backed S100/S100P support, never infer it from file names.

    Explicit target permits host preparation only. Execution separately checks
    actual hardware. External paths require an exact manifest asset identity.
    An optional image_size must equal the selected artifact's fixed geometry.
    """
    target = resolve_target(target, soc_name=soc_name, board_type=board_type)
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f'No published SigLIP support for {target}; use s100/s100p.')
    if submodel not in SUBMODELS:
        raise ValueError(f'Unknown SigLIP submodel: {submodel}')
    if model_path is not None and asset_id is None:
        raise ValueError('An external model-path requires the exact manifest asset-id.')
    assets = list_available_assets(target)
    if asset_id is None:
        variant = DEFAULT_VARIANT if variant is None else variant
        asset_id = f's:siglip:s100/bpu-siglip-{variant}.hbm'
    matches = [a for a in assets if a.reference == asset_id]
    if len(matches) != 1:
        raise ValueError(f'Unknown SigLIP asset-id: {asset_id}')
    asset = matches[0]
    actual = next(v for v in VARIANTS if asset.filename == f's100/bpu-siglip-{v}.hbm')
    if variant is not None and variant != actual:
        raise ValueError(f'Variant {variant!r} does not match asset {asset.reference}.')
    size = VARIANTS[actual][0]
    if image_size is not None and image_size != size:
        raise ValueError(f'image-size must be {size} for {actual}, got {image_size}.')
    path = Path(model_path).expanduser() if model_path is not None else SAMPLE_DIR/'model'/asset.filename
    return ModelSelection(asset, target, actual, path, submodel, size)


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    """Validate both-submodel presence and selected input/output metadata.

    pooler retains either documented (1,D) or published-table (1,1,D) shape;
    patch output must match (1,N,D), including N=729 for patch14/384.
    Output dtype is observed, numeric and preserved, not inferred as logits.
    """
    expected = resolve_selection(selection.target, variant=selection.variant,
        asset_id=selection.asset.reference, model_path=selection.model_path,
        submodel=selection.submodel, image_size=selection.image_size)
    if expected.asset != selection.asset:
        raise ValueError('Selection publication facts do not match manifest.')
    m = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if set(m.model_names) != set(SUBMODELS) or m.model_name != selection.submodel:
        raise MetadataMismatchError('Expected both SigLIP packed submodels and the explicit selected model.')
    size, dim, count = VARIANTS[selection.variant]
    input_shape = (1, 3, size, size)
    if m.input_names != ('_input_0',) or m.input_shapes.get('_input_0') != input_shape or m.input_dtypes.get('_input_0') != 'float32':
        raise MetadataMismatchError(f'SigLIP input must be _input_0 F32 {input_shape}.')
    allowed = ((1, dim), (1, 1, dim)) if selection.submodel == 'pooler_output' else ((1, count, dim),)
    shape = m.output_shapes.get('_output_0')
    dtype = m.output_dtypes.get('_output_0')
    if m.output_names != ('_output_0',) or shape not in allowed or dtype not in NUMERIC_DTYPES:
        raise MetadataMismatchError(f'SigLIP output must be _output_0 numeric {allowed}; got {shape}/{dtype}.')
    return ModelBinding(selection, m.model_name, input_shape, shape, dtype)
