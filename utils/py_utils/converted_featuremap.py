# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Bind registered F32 featuremap artifacts to the existing named-array runner.

Profiles belong to the sample, including exact compiled ABI and source graph
hash. Source-ONNX execution is a separate host reference, never an HBM result.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from utils.py_utils.assets import resolve_asset, sha256_file
from utils.py_utils.runtime_meta import RuntimeMetadata, MetadataMismatchError
from utils.py_utils.single_array_runner import NamedArrayRunner


@dataclass(frozen=True)
class Selection:
    target: str
    asset: object
    model_path: Path
    profile: dict


@dataclass(frozen=True)
class Binding:
    selection: Selection
    metadata: RuntimeMetadata

    @property
    def model_name(self):
        return self.metadata.model_name


def select(sample_dir, target, variant, *, asset_id=None, model_path=None):
    data=json.loads((Path(sample_dir)/'model/converted_featuremap.json').read_text())
    matches=[p for p in data['profiles']if p['target']==target and p['variant']==variant]
    if len(matches)!=1:raise ValueError('Unknown featuremap variant/target')
    profile=matches[0];asset=resolve_asset(profile['asset_id'])
    if asset.sha256!=profile['artifact_sha256']or (asset_id is not None and asset.reference!=asset_id):
        raise ValueError('Artifact differs from the registered profile')
    if asset.sample_id!=data['sample']or asset.group!='s' or not asset.filename.startswith(target+'/'):
        raise ValueError('Artifact belongs to another sample or target')
    if model_path is not None and asset_id is None:
        raise ValueError('--model-path requires the exact --asset-id')
    return Selection(target,asset,Path(model_path)if model_path is not None else Path(sample_dir)/'model'/asset.filename,profile)


def bind(selection, metadata):
    meta=metadata if isinstance(metadata,RuntimeMetadata)else RuntimeMetadata.from_mapping(metadata)
    profile=selection.profile
    if meta.model_names!=(profile['model_name'],)or meta.model_name!=profile['model_name']:
        raise MetadataMismatchError('Compiled model name differs from its measured ABI')
    for direction in ['input','output']:
        rows=profile['io'][direction+'s'];names=tuple(r['name']for r in rows)
        if getattr(meta,direction+'_names')!=names:
            raise MetadataMismatchError('Wrong '+direction+' tensor names/order')
        if getattr(meta,direction+'_shapes')!={r['name']:tuple(r['shape'])for r in rows}:
            raise MetadataMismatchError('Wrong '+direction+' tensor shapes')
        if getattr(meta,direction+'_dtypes')!={r['name']:r['dtype']for r in rows}:
            raise MetadataMismatchError('Wrong '+direction+' tensor dtypes')
    return Binding(selection,meta)


class Runner:
    """Select board execution or an exact, independently named source reference."""
    def __init__(self, selection, *, backend='board', source_onnx=None, runtime=None):
        self.selection=selection;self.backend=backend
        if backend=='source-onnx':
            import onnxruntime as ort
            if source_onnx is None or sha256_file(Path(source_onnx))!=selection.profile['source_onnx_sha256']:
                raise ValueError('Source ONNX differs from the accepted conversion')
            options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
            self.session=ort.InferenceSession(str(source_onnx),options,providers=['CPUExecutionProvider'])
            inputs=selection.profile['io']['inputs'];outputs=selection.profile['io']['outputs']
            if [(t.name,t.shape)for t in self.session.get_inputs()]!=[(t['name'],t['shape'])for t in inputs]:
                raise ValueError('Source input ABI differs')
            if [(t.name,t.shape)for t in self.session.get_outputs()]!=[(t['name'],t['shape'])for t in outputs]:
                raise ValueError('Source output ABI differs')
        elif backend=='board':
            self.runner=NamedArrayRunner(selection,binding_loader=bind,
                physical_inputs=lambda b:{r['name']:(tuple(r['shape']),r['dtype'])for r in b.selection.profile['io']['inputs']},
                task_name=selection.asset.sample_id,runtime=runtime)
        else:raise ValueError('Unknown backend')

    def __call__(self, tensor):
        inputs=self.selection.profile['io']['inputs']
        if len(inputs)!=1 or tensor.shape!=tuple(inputs[0]['shape'])or tensor.dtype!=np.float32 or not np.isfinite(tensor).all():
            raise ValueError('Expected the registered normalized F32 source input')
        feed={inputs[0]['name']:tensor}
        if self.backend=='board':return self.runner(feed)
        return {t.name:np.array(value,copy=True)for t,value in zip(self.session.get_outputs(),self.session.run(None,feed),strict=True)}
