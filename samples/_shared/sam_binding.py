# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Exact asset pairs and observed tensor protocols for two SAM consumers.

Source implementations cast native outputs to float32 without dequantization.
That transform belongs to each stage's post_process, never this binding. The
native dtype must still be known and every runtime array must match it. These
compatibility checks do not certify unobserved published HBM tensor metadata.
"""
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from samples._shared.assets import Asset, list_assets
from samples._shared.platforms import resolve_target
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata

SAMPLES = ('efficient_sam', 'mobile_sam')
TARGETS = ('x5', 's100', 's100p', 's600')
_MARCH = {'s100': ('nash-e', 'nashe'), 's100p': ('nash-m', 'nashm'), 's600': ('nash-p', 'nashp')}
_ROOT = Path(__file__).resolve().parents[2]
_NATIVE_OUTPUT_DTYPES = frozenset(('float16', 'float32', 'int8', 'uint8', 'int16', 'int32'))


def _freeze(value):
    """Detach nested metadata containers from mutable SDK-owned mappings."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    return value


@dataclass(frozen=True)
class ModelSelection:
    """A pair from one sample and one target; a path is not proof of identity."""
    sample: str
    target: str
    encoder_asset: Asset
    decoder_asset: Asset
    encoder_model_path: Path
    decoder_model_path: Path


@dataclass(frozen=True)
class StageBinding:
    """Source protocol with exact metadata observed when this stage was loaded."""
    sample: str
    target: str
    stage: str
    metadata: RuntimeMetadata

    @property
    def model_name(self):
        return self.metadata.model_name

    @property
    def input_names(self):
        return self.metadata.input_names

    @property
    def output_names(self):
        return self.metadata.output_names


@dataclass(frozen=True)
class ModelBinding:
    """Both stages must bind successfully before a pipeline may execute."""
    selection: ModelSelection
    encoder: StageBinding
    decoder: StageBinding


def _filenames(sample, target):
    if sample not in SAMPLES:
        raise ValueError(f'Unknown SAM sample: {sample!r}.')
    if target not in TARGETS:
        raise ValueError(f'Unknown SAM target: {target!r}.')
    if target == 'x5':
        if sample == 'efficient_sam':
            return ('efficient_sam_vitt_encoder_512x512_default_none.bin',
                    'efficient_sam_vitt_decoder_fixedprompt_512_default.bin')
        return ('mobile_sam_image_encoder_norm_512x512_allint16.bin',
                'mobile_sam_decoder_512_box_default.bin')
    march, suffix = _MARCH[target]
    encoder = 'efficient_sam_vitt_encoder' if sample == 'efficient_sam' else 'mobile_sam_image_encoder_norm'
    decoder = 'efficient_sam_vitt_decoder' if sample == 'efficient_sam' else 'mobile_sam_decoder'
    return (f'{march}/{encoder}_512x512_{suffix}.hbm', f'{march}/{decoder}_512_{suffix}.hbm')


def list_available_assets(sample, target='auto'):
    """List exact pairs; auto means all published targets without board reads."""
    if target in (None, 'auto'):
        return tuple(asset for item in TARGETS for asset in list_available_assets(sample, item))
    names = _filenames(sample, target)
    group, fmt = ('x5', 'bin') if target == 'x5' else ('s', 'hbm')
    rows = list_assets(group, sample)
    result = []
    for name in names:
        matches = [a for a in rows if a.filename == name and a.format == fmt]
        if len(matches) != 1:
            raise ValueError(f'Manifest has no unique {sample}/{target} asset {name!r}.')
        result.append(matches[0])
    return tuple(result)


def resolve_selection(sample, target='auto', *, encoder_model_path=None,
                      decoder_model_path=None, encoder_asset_id=None,
                      decoder_asset_id=None, sample_dir=None):
    """Resolve exact manifest stage identities without loading or downloading.

    External paths require the corresponding qualified asset ID. A supplied ID
    must match sample, target and stage. Unknown publisher hashes remain unknown;
    this selection does not authenticate an externally supplied file's bytes.
    """
    target = resolve_target(target)
    assets = list_available_assets(sample, target)
    directory = Path(sample_dir) if sample_dir is not None else _ROOT/'samples'/'vision'/sample
    paths = []
    for stage, asset, explicit, identity in zip(
        ('encoder', 'decoder'), assets,
        (encoder_model_path, decoder_model_path), (encoder_asset_id, decoder_asset_id)
    ):
        if explicit is not None and identity is None:
            raise ValueError(f'External {stage} path requires its exact {stage}-asset-id.')
        if identity is not None and identity != asset.reference:
            raise ValueError(f'{stage} expected asset-id {asset.reference}, got {identity!r}.')
        paths.append(Path(explicit).expanduser() if explicit is not None else directory/'model'/asset.filename)
    return ModelSelection(sample, target, *assets, *paths)


def bind_stage(sample, target, stage, metadata):
    """Validate actual stage metadata against a bounded source protocol.

    Encoder input is F32[1,3,512,512]; embedding is [1,256,32,32]. Decoder
    input is always F32. X5 MobileSAM permits the source runtime box shape
    [1,4,1,1] and exported [1,4]; S permits only [1,4]. Decoder outputs have
    exactly three candidates, batch one. X5 shapes are fixed by its model
    documentation. S spatial dimensions are read from metadata, not guessed.
    """
    _filenames(sample, target)
    if stage not in ('encoder', 'decoder'):
        raise MetadataMismatchError(f'Unknown SAM stage {stage!r}.')
    meta = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    prefix = f'{sample} {target} {stage}'
    if not meta.model_name or meta.model_names != (meta.model_name,):
        raise MetadataMismatchError(f'{prefix}: each artifact must contain exactly one model.')
    image_input = 'batched_images' if sample == 'efficient_sam' else 'normalized_images'
    shapes = {image_input: (1, 3, 512, 512)} if stage == 'encoder' else {'image_embeddings': (1, 256, 32, 32)}
    if stage == 'decoder' and sample == 'mobile_sam':
        permitted = ((1, 4), (1, 4, 1, 1)) if target == 'x5' else ((1, 4),)
        shape = meta.input_shapes.get('boxes')
        if shape not in permitted:
            raise MetadataMismatchError(f'{prefix}: boxes shape {shape!r} is outside {permitted!r}.')
        shapes['boxes'] = shape
    if len(meta.input_names) != len(shapes) or set(meta.input_names) != set(shapes):
        raise MetadataMismatchError(f'{prefix}: expected inputs {tuple(shapes)!r}.')
    for name, shape in shapes.items():
        if meta.input_shapes.get(name) != shape or meta.input_dtypes.get(name) != 'float32':
            raise MetadataMismatchError(f'{prefix}: {name} must be float32{shape}.')
    names = ('image_embeddings',) if stage == 'encoder' else ('low_res_masks', 'iou_predictions')
    if len(meta.output_names) != len(names) or set(meta.output_names) != set(names):
        raise MetadataMismatchError(f'{prefix}: expected outputs {names!r}.')
    for name in names:
        if meta.output_dtypes.get(name) not in _NATIVE_OUTPUT_DTYPES:
            raise MetadataMismatchError(f'{prefix}: unknown or unsupported native dtype for {name!r}.')
    if stage == 'encoder':
        if meta.output_shapes.get('image_embeddings') != (1, 256, 32, 32):
            raise MetadataMismatchError(f'{prefix}: embedding must be [1,256,32,32].')
    else:
        mask = meta.output_shapes.get('low_res_masks', ())
        iou = meta.output_shapes.get('iou_predictions', ())
        if target == 'x5':
            valid = mask == (1, 3, 128, 128) and iou == (1, 3, 1, 1)
        else:
            valid = (len(mask) == 4 and mask[:2] == (1, 3)
                     and all(type(d) is int and d > 0 for d in mask[2:])
                     and iou in ((1, 3), (1, 3, 1, 1)))
        if not valid:
            raise MetadataMismatchError(f'{prefix}: unsupported mask/IoU shape {mask!r}/{iou!r}.')
    meta = replace(meta, **{name: _freeze(getattr(meta, name)) for name in (
        'input_shapes', 'input_dtypes', 'output_shapes', 'output_dtypes',
        'input_strides', 'output_strides', 'output_quants', 'output_semantics')})
    return StageBinding(sample, target, stage, meta)


def bind_model(selection, encoder_metadata, decoder_metadata):
    """Recheck publication facts and bind both stage protocols atomically."""
    expected = resolve_selection(selection.sample, selection.target,
        encoder_model_path=selection.encoder_model_path,
        decoder_model_path=selection.decoder_model_path,
        encoder_asset_id=selection.encoder_asset.reference,
        decoder_asset_id=selection.decoder_asset.reference)
    if expected != selection:
        raise MetadataMismatchError('SAM selection differs from exact manifest publication facts.')
    return ModelBinding(selection,
        bind_stage(selection.sample, selection.target, 'encoder', encoder_metadata),
        bind_stage(selection.sample, selection.target, 'decoder', decoder_metadata))


def validate_tensors(binding, tensors, *, outputs=False):
    """Validate names, exact shape/dtype and finite values, preserving arrays.

    No cast, reshape, activation, dequantization or writable-buffer copy occurs.
    Stage post_process owns numeric transformation and result ownership.
    """
    meta = binding.metadata
    names = meta.output_names if outputs else meta.input_names
    shapes = meta.output_shapes if outputs else meta.input_shapes
    dtypes = meta.output_dtypes if outputs else meta.input_dtypes
    if not isinstance(tensors, Mapping) or set(tensors) != set(names):
        raise MetadataMismatchError(f'{binding.stage}: expected exactly tensors {names!r}.')
    for name in names:
        value = tensors[name]
        if (not isinstance(value, np.ndarray) or value.shape != shapes[name]
                or value.dtype != np.dtype(dtypes[name]) or not np.isfinite(value).all()):
            raise MetadataMismatchError(f'{binding.stage}: {name!r} differs from finite native '
                                        f'{dtypes[name]}{shapes[name]}.')
    return dict(tensors)
