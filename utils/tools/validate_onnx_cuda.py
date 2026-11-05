"""Check every source ONNX output on CPU/CUDA with frozen real input tensors."""
from pathlib import Path
import argparse
import hashlib
import json
import os

parser=argparse.ArgumentParser()
parser.add_argument('--case',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
import numpy as np
import onnxruntime as ort


def identity(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':p.stat().st_size}


case=json.loads(args.case.read_text());onnx=case['onnx'];assert identity(onnx['path'])==onnx
assert 'CUDAExecutionProvider' in ort.get_available_providers()
options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
cpu=ort.InferenceSession(onnx['path'],sess_options=options,providers=['CPUExecutionProvider'])
gpu=ort.InferenceSession(onnx['path'],sess_options=options,
    providers=[('CUDAExecutionProvider',{'use_tf32':0}),'CPUExecutionProvider'])
assert gpu.get_providers()[0]=='CUDAExecutionProvider'
inputs=cpu.get_inputs();outputs=cpu.get_outputs();checks=[];samples=[]
for row in case['real_inputs']:
    feed={}
    for meta in inputs:
        item=row[meta.name];assert identity(item['path'])==item
        value=np.load(item['path'],allow_pickle=False)
        assert meta.type=='tensor(float)' and value.dtype==np.float32 and list(value.shape)==meta.shape
        assert np.isfinite(value).all();feed[meta.name]=value
    samples.append((Path(next(iter(row.values()))['path']).name,feed))
generator=np.random.default_rng(0);random={}
for meta in inputs:
    reference=samples[0][1][meta.name]
    if meta.name=='boxes':value=np.array([[24,32,256,384]],np.float32)
    elif reference.min()>=0 and reference.max()<=1:value=generator.uniform(0,1,meta.shape).astype(np.float32)
    else:value=generator.normal(size=meta.shape).astype(np.float32)
    random[meta.name]=value
samples.append(('random-seed0',random))
failed=False
for name,feed in samples:
    expected=cpu.run(None,feed);actual=gpu.run(None,feed);comparisons=[]
    assert len(expected)==len(actual)==len(outputs)
    for meta,a,b in zip(outputs,expected,actual):
        finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
        violations=int(np.count_nonzero(~np.isclose(b,a,atol=1e-4,rtol=1e-3)))
        passed=finite and violations==0 and a.shape==b.shape and a.dtype==b.dtype
        failed|=not passed
        comparisons.append({'output':meta.name,'shape':list(a.shape),'dtype':str(a.dtype),
            'finite':finite,'max_abs':float(np.max(np.abs(a-b))),'violations':violations,
            'status':'passed' if passed else 'failed'})
    checks.append({'case':name,'outputs':comparisons,'status':'passed' if all(v['status']=='passed' for v in comparisons) else 'failed'})
result={'schema_version':1,'status':'failed' if failed else 'passed','kind':'source-onnx-cpu-cuda-parity',
    'onnx':onnx,'calibration_manifest':case['calibration_manifest'],'real_inputs':case['real_inputs'],
    'atol':1e-4,'rtol':1e-3,'TF32':False,'NVIDIA_TF32_OVERRIDE':os.getenv('NVIDIA_TF32_OVERRIDE'),
    'onnxruntime':ort.__version__,'providers':{'cpu':cpu.get_providers(),'cuda':gpu.get_providers()},
    'toolchain_image_id':case['toolchain_image_id'],'producer':identity(__file__),'cases':checks,
    'scope':'Source graph outputs on frozen real tensors plus seed-0 input; no PTQ, model changes or board execution',
    'board':'not-run'}
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'onnx':onnx['sha256'],'cases':[{k:r[k] for k in ('case','status')} for r in checks]}),flush=True)
raise SystemExit(1 if failed else 0)
