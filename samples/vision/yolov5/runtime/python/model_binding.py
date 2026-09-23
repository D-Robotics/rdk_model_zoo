# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Exact published YOLOv5/ByteTrack detector identities and tensor contracts."""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import math
from samples._shared.assets import Asset, list_assets
from samples._shared.platforms import resolve_target
from samples._shared.runtime_meta import RuntimeMetadata, MetadataMismatchError

SAMPLE_DIR = Path(__file__).resolve().parents[2]
X5_VARIANTS = ('n-v7.0','s-v2.0','m-v2.0','l-v2.0','x-v2.0','s-v7.0','m-v7.0','l-v7.0','x-v7.0')
STRIDES = (8,16,32)
ANCHORS = (10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326)


def _x5_filename(variant):
    size,tag=variant.split('-v');return f'yolov5{size}_tag_v{tag}_detect_640x640_bayese_nv12.bin'


@dataclass(frozen=True)
class ModelSelection:
    asset: Asset
    target: str
    model_path: Path
    variant: str
    consumer: str = 'yolov5'


@dataclass(frozen=True)
class Quantization:
    quant_type: str
    scale: object
    zero_point: object
    axis: int


@dataclass(frozen=True)
class ModelBinding:
    selection: ModelSelection
    model_name: str
    input_names: tuple
    input_shapes: object
    output_names: tuple
    output_shapes: object
    output_dtypes: object
    output_quants: object
    input_size: int
    output_transform: str
    resize_type: int
    activation: str = 'sigmoid_logits'


def list_available_assets(target=None, *, consumer='yolov5'):
    """List finite source-published assets; does not inspect hardware or fetch data."""
    if consumer not in ('yolov5','bytetrack'):raise ValueError('Unknown detector consumer.')
    if target not in (None,'auto','x5','s100','s100p','s600'):raise ValueError('Unknown target.')
    groups=('x5','s') if consumer=='yolov5' else ('s',)
    assets=[]
    for group in groups:
        rows=list_assets(group,consumer)
        expected=({_x5_filename(v) for v in X5_VARIANTS} if group=='x5' else
                  {f'{t}/yolov5x_672x672_nv12.hbm' for t in (('s100','s600') if consumer=='yolov5' else ('s100','s100p','s600'))})
        if {a.filename for a in rows}!=expected or any(a.format!=('bin' if group=='x5' else 'hbm') for a in rows):
            raise ValueError(f'{consumer} publication changed; review its finite contracts.')
        assets.extend(a for a in rows if target in (None,'auto') or (group=='x5' and target=='x5') or (group=='s' and a.filename.startswith(target+'/')))
    return tuple(assets)


def resolve_selection(target='auto', *, variant=None, asset_id=None, model_path=None, consumer='yolov5', soc_name=None, board_type=None):
    """Resolve an exact target/variant. Custom paths require the matching asset ID."""
    target=resolve_target(target,soc_name=soc_name,board_type=board_type)
    assets=list_available_assets(target,consumer=consumer)
    if not assets:raise ValueError(f'No published {consumer} asset for {target}.')
    if model_path is not None and asset_id is None:raise ValueError('External model-path requires an exact asset-id.')
    if asset_id is not None:
        matches=[a for a in assets if a.reference==asset_id]
        if len(matches)!=1:raise ValueError(f'Unknown or mismatched {consumer} asset-id: {asset_id}.')
        asset=matches[0]
        inferred=next((v for v in X5_VARIANTS if _x5_filename(v)==asset.filename),'x-672')
        if variant is not None and variant!=inferred:raise ValueError('variant and asset-id mismatch.')
        variant=inferred
    else:
        variant=variant or ('n-v7.0' if target=='x5' else 'x-672')
        if target=='x5' and variant in X5_VARIANTS:filename=_x5_filename(variant)
        elif target!='x5' and variant=='x-672':filename=f'{target}/yolov5x_672x672_nv12.hbm'
        else:raise ValueError(f'Unsupported {target} variant: {variant}.')
        asset=next(a for a in assets if a.filename==filename)
    root=SAMPLE_DIR if consumer=='yolov5' else SAMPLE_DIR.parent/'bytetrack'
    return ModelSelection(asset,target,Path(model_path).expanduser() if model_path is not None else root/'model'/asset.filename,variant,consumer)


def _quant_snapshot(info, shape, dtype):
    import numpy as np
    if info is None:raise MetadataMismatchError('S output requires its source quantization descriptor.')
    q=getattr(info,'quant_type',info);kind=str(getattr(q,'name',q))
    if kind not in ('SCALE','1','NONE','0'):raise MetadataMismatchError('Unsupported quantization descriptor kind.')
    if dtype!='float32' and kind not in ('SCALE','1'):raise MetadataMismatchError('Integer output requires SCALE quantization.')
    scale=tuple(float(x) for x in np.asarray(getattr(info,'scale',())).reshape(-1))
    zero=tuple(float(x) for x in np.asarray(getattr(info,'zero_point',())).reshape(-1))
    axis=int(getattr(info,'axis',0))
    if kind in ('SCALE','1'):
        if not scale or not all(math.isfinite(x) and x>0 for x in scale) or not all(math.isfinite(x) for x in zero):raise MetadataMismatchError('Invalid quantization scale/zero point.')
        if len(scale)>1 and (not -len(shape)<=axis<len(shape) or len(scale)!=shape[axis] or len(zero) not in (0,1,len(scale))):raise MetadataMismatchError('Quantization channels/axis mismatch.')
    def frozen_array(value):
        a=np.asarray(value)
        return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)
    return Quantization(kind,frozen_array(getattr(info,'scale',())),frozen_array(getattr(info,'zero_point',())),axis)


def bind_model(selection, metadata):
    """Validate actual native shapes/dtypes; never infer an HBM layout from a name."""
    check=resolve_selection(selection.target,variant=selection.variant,asset_id=selection.asset.reference,model_path=selection.model_path,consumer=selection.consumer)
    if check.asset!=selection.asset:raise ValueError('Manifest publication facts changed.')
    m=metadata if isinstance(metadata,RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if m.model_names!=(m.model_name,):raise MetadataMismatchError('YOLOv5 artifact must contain exactly one model.')
    size=640 if selection.target=='x5' else 672
    expected=((1,3,size,size),) if selection.target=='x5' else ((1,size,size,1),(1,size//2,size//2,2))
    if len(m.input_names)!=len(expected) or tuple(m.input_shapes.get(n) for n in m.input_names)!=expected:raise MetadataMismatchError('YOLOv5 logical input shapes/order mismatch.')
    if any(m.input_dtypes.get(n) not in ('uint8','nv12') for n in m.input_names):raise MetadataMismatchError('YOLOv5 inputs require uint8/NV12 metadata.')
    if len(set(m.output_names))!=3:raise MetadataMismatchError('YOLOv5 requires three distinct detection heads.')
    ordered=[];quants={};dtypes={}
    for stride in STRIDES:
        h=size//stride
        matches=[n for n in m.output_names if m.output_shapes.get(n) in ((1,h,h,255),(1,h,h,3,85))]
        if len(matches)!=1:raise MetadataMismatchError(f'No unique stride-{stride} 3-anchor/80-class head.')
        name=matches[0];ordered.append(name);dtype=m.output_dtypes.get(name)
        allowed=('float32',) if selection.target=='x5' else ('float32','int8','uint8','int16','int32')
        if dtype not in allowed:raise MetadataMismatchError(f'Unsupported native output dtype {dtype!r}.')
        dtypes[name]=dtype
        if selection.target!='x5':quants[name]=_quant_snapshot(m.output_quants.get(name),m.output_shapes[name],dtype)
    freeze=lambda x:MappingProxyType(dict(x))
    return ModelBinding(selection,m.model_name,m.input_names,freeze(m.input_shapes),tuple(ordered),freeze(m.output_shapes),freeze(dtypes),freeze(quants),size,'raw_f32' if selection.target=='x5' else 'dequant',0 if selection.target=='x5' else 1)


def validate_tensors(binding,tensors,*,outputs=False):
    """Validate flat tensors without casting, reshaping or replacing array values."""
    import numpy as np
    from collections.abc import Mapping
    names=binding.output_names if outputs else binding.input_names
    if not isinstance(tensors,Mapping) or set(tensors)!=set(names):raise MetadataMismatchError('Tensor names do not match binding.')
    for n in names:
        value=tensors[n]
        shape=binding.output_shapes[n] if outputs else ((binding.input_size**2*3//2,) if binding.selection.target=='x5' else binding.input_shapes[n])
        dtype=binding.output_dtypes[n] if outputs else 'uint8'
        if not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype!=np.dtype(dtype):raise MetadataMismatchError(f'{n}: expected native {shape}/{dtype}.')
        if not np.isfinite(value).all():raise MetadataMismatchError(f'{n}: non-finite tensor.')
    return {n:tensors[n] for n in names}
