# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Exact S100/S600 asset selection and split-NV12 segmentation contracts."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from samples._shared.assets import Asset, list_assets
from samples._shared.platforms import resolve_target
from samples._shared.quantization import validate_scale_quantization
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata

SAMPLE_DIR = Path(__file__).resolve().parents[2]
SUPPORTED_TARGETS = ('x5', 's100', 's100p', 's600')
FILE_NAME = 'unet_mobilenet_1024x2048_nv12.hbm'


@dataclass(frozen=True)
class ModelSelection:
    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


@dataclass(frozen=True)
class ModelBinding:
    selection: ModelSelection
    metadata: RuntimeMetadata
    y_name: str
    uv_name: str
    output_name: str
    input_height: int = 1024
    input_width: int = 2048

    @property
    def model_name(self):
        return self.metadata.model_name


def list_available_assets(target=None):
    rows = tuple(list_assets('s', 'unetmobilenet'))
    if target in (None, 'auto'):
        return rows
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f'Unknown target {target!r}')
    return tuple(a for a in rows if a.filename.startswith(target+'/'))


def resolve_selection(target='auto', *, asset_id=None, model_path=None):
    rows = list_available_assets()
    if asset_id is not None:
        matches = [asset for asset in rows if asset.reference == asset_id]
        if len(matches) != 1:
            raise ValueError(f'Unknown UnetMobileNet asset-id {asset_id!r}')
        inferred = matches[0].filename.split('/')[0]
        if target in (None, 'auto'):
            target = inferred
        elif target != inferred:
            raise ValueError('target and asset-id select different artifacts')
    key = resolve_target(target) if target in (None, "auto") else target
    if key not in ('s100', 's600'):
        raise ValueError(f'UnetMobileNet has no published asset for {key}')
    matches = [asset for asset in rows if asset.filename == f'{key}/{FILE_NAME}']
    if len(matches) != 1:
        raise ValueError(f'Expected one UnetMobileNet asset for {key}')
    if model_path is not None and asset_id is None:
        raise ValueError('External model paths require the exact --asset-id')
    asset = matches[0]
    path = Path(model_path).expanduser() if model_path is not None else SAMPLE_DIR/'model'/asset.filename
    return ModelSelection(key, asset, path, model_path is not None)


def bind_model(selection, metadata: RuntimeMetadata | Mapping[str, Any]):
    resolved = resolve_selection(selection.target, asset_id=selection.asset.reference,
                                 model_path=selection.model_path if selection.explicit_model_path else None)
    if selection != resolved:
        raise ValueError('ModelSelection does not match the manifest identity and path')
    meta = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if meta.model_names != (meta.model_name,) or len(meta.input_names) != 2 or len(meta.output_names) != 1:
        raise MetadataMismatchError('UnetMobileNet requires one model, split Y/UV and one logits tensor')
    y = [name for name in meta.input_names if meta.input_shapes.get(name) == (1, 1024, 2048, 1)]
    uv = [name for name in meta.input_names if meta.input_shapes.get(name) == (1, 512, 1024, 2)]
    if len(y) != 1 or len(uv) != 1 or any(meta.input_dtypes.get(name) != 'uint8' for name in meta.input_names):
        raise MetadataMismatchError('Expected uint8 Y [1,1024,2048,1] and UV [1,512,1024,2]')
    output = meta.output_names[0]
    shape = meta.output_shapes.get(output, ())
    if len(shape) != 4 or shape[0] != 1 or shape[-1] != 19 or min(shape[1:3]) <= 0:
        raise MetadataMismatchError('Expected NHWC [1,H,W,19] segmentation logits')
    dtype = meta.output_dtypes.get(output)
    if dtype not in ('int32', 'float32'):
        raise MetadataMismatchError('Expected source int32 or explicit float32 logits')
    if dtype == 'int32':
        info = meta.output_quants.get(output)
        kind = getattr(info, 'quant_type', None)
        kind = str(getattr(kind, 'name', kind))
        if kind in ('SCALE', '1'):
            if not hasattr(info, 'zero_point'):
                raise MetadataMismatchError('SCALE requires zero_point metadata (empty means symmetric)')
            validate_scale_quantization(info, shape)
        elif kind not in ('NONE', '0'):
            raise MetadataMismatchError('int32 logits require explicit NONE or valid SCALE metadata')
    return ModelBinding(selection, meta, y[0], uv[0], output)
