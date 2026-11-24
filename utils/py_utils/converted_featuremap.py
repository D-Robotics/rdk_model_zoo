# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Bind registered F32 featuremap artifacts to the existing named-array runner.

Profiles belong to the sample, including exact compiled ABI and source graph
hash. Source-ONNX execution is a separate host reference, never an HBM result.
"""
from dataclasses import dataclass
from collections.abc import Mapping
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
            if selection.profile.get('ort_optimization_level') == 'basic':
                options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            self.session=ort.InferenceSession(str(source_onnx),options,providers=['CPUExecutionProvider'])
            inputs=selection.profile['io']['inputs'];outputs=selection.profile['io']['outputs']
            dtype={'float32':'tensor(float)','int64':'tensor(int64)'}
            if [(t.name,t.shape,t.type)for t in self.session.get_inputs()]!=[(t['name'],t['shape'],dtype[t['dtype']])for t in inputs]:
                raise ValueError('Source input ABI differs')
            if [(t.name,t.shape,t.type)for t in self.session.get_outputs()]!=[(t['name'],t['shape'],dtype[t['dtype']])for t in outputs]:
                raise ValueError('Source output ABI differs')
        elif backend=='board':
            self.runner=NamedArrayRunner(selection,binding_loader=bind,
                physical_inputs=lambda b:{r['name']:(tuple(r['shape']),r['dtype'])for r in b.selection.profile['io']['inputs']},
                task_name=selection.asset.sample_id,runtime=runtime)
        else:raise ValueError('Unknown backend')

    def __call__(self, tensor):
        inputs=self.selection.profile['io']['inputs']
        if isinstance(tensor, Mapping):
            feed=dict(tensor)
        elif len(inputs)==1:
            feed={inputs[0]['name']:tensor}
        else:
            raise ValueError('Multiple inputs require their exact registered names')
        if set(feed)!={row['name']for row in inputs}:
            raise ValueError('Input names differ from the registered source ABI')
        for row in inputs:
            value=feed[row['name']]
            if (not isinstance(value,np.ndarray) or value.shape!=tuple(row['shape'])
                    or value.dtype!=np.dtype(row['dtype']) or not np.isfinite(value).all()):
                raise ValueError('Expected the registered source input shape and dtype')
        if self.backend=='board':return self.runner(feed)
        return {t.name:np.array(value,copy=True)for t,value in zip(self.session.get_outputs(),self.session.run(None,feed),strict=True)}


def main(argv=None):
    """Explicitly prepare one registered featuremap artifact without a board SDK."""
    import argparse
    from utils.py_utils.assets import download_asset
    p=argparse.ArgumentParser(description=main.__doc__)
    p.add_argument('--sample',choices=['vit','fcos','lanenet','unet','3dresnet','siglip','asr'],required=True)
    p.add_argument('--target',choices=['s100'],default='s100')
    p.add_argument('--variant',required=True);p.add_argument('--asset-id')
    p.add_argument('--output-dir',type=Path);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    domain='speech'if a.sample=='asr'else'vision'
    sample=Path(__file__).resolve().parents[2]/'samples'/domain/a.sample
    selection=select(sample,a.target,a.variant,asset_id=a.asset_id)
    destination=a.output_dir/selection.asset.filename if a.output_dir else selection.model_path
    if not a.dry_run:download_asset(selection.asset,destination)
    print(json.dumps({'asset_id':selection.asset.reference,'path':str(destination),
                      'sha256':selection.asset.sha256,'download':'not-run'if a.dry_run else'verified'}))
    return 0


if __name__=='__main__':raise SystemExit(main())
