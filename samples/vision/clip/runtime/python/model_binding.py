# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Published X5 encoder pair and its two runtime metadata contracts.

Protocol provenance: ac115717197920355fc390bb04299b20e6436864,
CLIP conversion README and runtime. Tensor names are obtained from actual
metadata, since the source's protocol table gives conceptual names only.
"""
from dataclasses import dataclass
from pathlib import Path
from samples._shared.assets import Asset, list_assets
from samples._shared.platforms import resolve_target
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata

SAMPLE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSelection:
    target: str
    image_asset: Asset
    text_asset: Asset
    image_model_path: Path
    text_model_path: Path


@dataclass(frozen=True)
class ModelBinding:
    selection: ModelSelection
    image_model_name: str
    image_input_name: str
    image_output_name: str
    text_input_name: str
    text_output_name: str
    text_batch_size: int | None


def list_available_assets(target=None):
    """List the exact pair; other known boards have no published CLIP pair."""
    if target in ('s100', 's100p', 's600'):
        return ()
    if target not in (None, 'auto', 'x5'):
        raise ValueError(f'Unknown target: {target}')
    assets = tuple(list_assets('x5', 'clip'))
    if len(assets) != 2 or {(a.filename, a.format) for a in assets} != {
        ('img_encoder.bin', 'bin'), ('text_encoder.onnx', 'onnx')
    }:
        raise ValueError('CLIP publication changed; review the encoder pair first.')
    return assets


def resolve_selection(target='auto', *, image_asset_id=None, text_asset_id=None,
                      image_model_path=None, text_model_path=None):
    """Select both encoders; external paths require the corresponding exact ID."""
    target = resolve_target(target)
    if target != 'x5':
        raise ValueError(f'No published CLIP encoder pair for {target}.')
    assets = {a.filename: a for a in list_available_assets(target)}
    selected = []
    paths = []
    for filename, asset_id, path in (
        ('img_encoder.bin', image_asset_id, image_model_path),
        ('text_encoder.onnx', text_asset_id, text_model_path),
    ):
        asset = assets[filename]
        if path is not None and asset_id is None:
            raise ValueError(f'External {filename} path requires its exact asset-id.')
        if asset_id is not None and asset_id != asset.reference:
            raise ValueError(f'Expected exact asset-id {asset.reference}, got {asset_id!r}.')
        selected.append(asset)
        paths.append(Path(path).expanduser() if path is not None else SAMPLE_DIR/'model'/filename)
    return ModelSelection(target, *selected, *paths)


def _batch_dimension(dim):
    if dim is None or isinstance(dim, str):
        return None
    if type(dim) is int and dim > 0:
        return dim
    raise MetadataMismatchError(f'Invalid ONNX batch dimension {dim!r}.')


def bind_model(selection, image_metadata, text_session):
    """Validate F32 image features and I32-token/F32-feature ONNX tensors."""
    expected = resolve_selection(selection.target, image_asset_id=selection.image_asset.reference,
                                 text_asset_id=selection.text_asset.reference,
                                 image_model_path=selection.image_model_path,
                                 text_model_path=selection.text_model_path)
    if expected != selection:
        raise MetadataMismatchError('Encoder selection differs from publication facts.')
    facts = image_metadata if isinstance(image_metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(image_metadata)
    if len(facts.input_names) != 1 or len(facts.output_names) != 1:
        raise MetadataMismatchError('CLIP image encoder requires one input and one output.')
    inp, out = facts.input_names[0], facts.output_names[0]
    if facts.input_shapes.get(inp) != (1, 3, 224, 224) or facts.input_dtypes.get(inp) != 'float32':
        raise MetadataMismatchError('CLIP image input must be F32[1,3,224,224].')
    if facts.output_shapes.get(out) != (1, 512) or facts.output_dtypes.get(out) != 'float32':
        raise MetadataMismatchError('CLIP image output must be F32[1,512].')
    inputs, outputs = text_session.get_inputs(), text_session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise MetadataMismatchError('CLIP text encoder requires one input and one output.')
    text_in, text_out = inputs[0], outputs[0]
    for desc, dtype, width in ((text_in, 'tensor(int32)', 77), (text_out, 'tensor(float)', 512)):
        if not isinstance(desc.name, str) or not desc.name or desc.type != dtype or len(desc.shape) != 2 or desc.shape[1] != width:
            raise MetadataMismatchError(f'Invalid CLIP text tensor metadata: {desc.name!r}.')
    batches = [_batch_dimension(text_in.shape[0]), _batch_dimension(text_out.shape[0])]
    fixed = {size for size in batches if size is not None}
    if len(fixed) > 1:
        raise MetadataMismatchError('CLIP text input/output batch metadata conflicts.')
    return ModelBinding(selection, facts.model_name, inp, out, text_in.name,
                        text_out.name, next(iter(fixed), None))
