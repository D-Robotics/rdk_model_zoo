"""Freeze host CPU ONNX outputs, then compare the cached CPU SDK exactly."""
from pathlib import Path
import argparse
import hashlib
import json

parser=argparse.ArgumentParser()
parser.add_argument('--case',type=Path,required=True)
parser.add_argument('--reference',type=Path,required=True)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
import numpy as np
import onnxruntime as ort


def identity(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':p.stat().st_size}


case=json.loads(args.case.read_text());assert identity(case['onnx']['path'])==case['onnx']
manifest=case['calibration_manifest'];assert identity(manifest['path'])['sha256']==manifest['sha256']
options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
session=ort.InferenceSession(case['onnx']['path'],sess_options=options,providers=['CPUExecutionProvider'])
inputs=session.get_inputs();outputs=session.get_outputs();samples=[]
assert len(case['real_inputs'])==3 and inputs and outputs
for row in case['real_inputs']:
    assert set(row)=={meta.name for meta in inputs}
    feed={}
    for meta in inputs:
        item=row[meta.name];assert identity(item['path'])==item
        x=np.load(item['path'],allow_pickle=False)
        assert meta.type=='tensor(float)' and x.dtype==np.float32 and list(x.shape)==meta.shape and np.isfinite(x).all()
        feed[meta.name]=x
    samples.append((Path(next(iter(row.values()))['path']).name,feed))
generator=np.random.default_rng(0);random={}
for meta in inputs:
    x=samples[0][1][meta.name]
    random[meta.name]=(generator.uniform(0,1,meta.shape) if x.min()>=0 and x.max()<=1 else generator.normal(size=meta.shape)).astype(np.float32)
samples.append(('random-seed0',random))
producer=identity(__file__);checks=[]
if args.output is None:
    args.reference.mkdir(parents=True,exist_ok=False);records=[]
    for index,(name,feed) in enumerate(samples):
        values=session.run(None,feed);assert len(values)==len(outputs) and all(np.isfinite(x).all() for x in values)
        path=args.reference/(str(index)+'.npz');np.savez(path,**{m.name:x for m,x in zip(outputs,values)})
        records.append({'case':name,'outputs':identity(path)})
    record={'status':'frozen','onnx':case['onnx'],'real_inputs':case['real_inputs'],'cases':records,
        'provider':session.get_providers(),'onnxruntime':ort.__version__,'producer':producer}
    (args.reference/'reference.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'frozen','onnxruntime':ort.__version__,'cases':len(records)}),flush=True)
else:
    reference_file=args.reference/'reference.json';reference=json.loads(reference_file.read_text())
    assert reference['onnx']==case['onnx'] and reference['real_inputs']==case['real_inputs']
    assert reference.get('status')=='frozen'
    assert len(reference['cases'])==len(samples)
    assert [record['case'] for record in reference['cases']]==[name for name,feed in samples]
    failed=False
    for (name,feed),record in zip(samples,reference['cases']):
        assert name==record['case'] and identity(record['outputs']['path'])==record['outputs']
        actual=session.run(None,feed);assert len(actual)==len(outputs);comparisons=[]
        with np.load(record['outputs']['path'],allow_pickle=False) as expected:
            assert set(expected.files)=={m.name for m in outputs}
            for meta,value in zip(outputs,actual):
                baseline=expected[meta.name];assert value.shape==baseline.shape and value.dtype==baseline.dtype
                finite=bool(np.isfinite(value).all());violations=int(np.count_nonzero(~np.isclose(value,baseline,atol=1e-4,rtol=1e-3)))
                passed=finite and violations==0;failed|=not passed
                comparisons.append({'output':meta.name,'finite':finite,'max_abs':float(np.max(np.abs(value-baseline))),
                    'violations':violations,'status':'passed' if passed else 'failed'})
        checks.append({'case':name,'outputs':comparisons,'status':'passed' if all(v['status']=='passed' for v in comparisons) else 'failed'})
    report={'schema_version':1,'kind':'cpu-onnx-reference-parity','status':'failed' if failed else 'passed',
        'onnx':case['onnx'],'calibration_manifest':case['calibration_manifest'],'real_inputs':case['real_inputs'],
        'reference':identity(reference_file),'reference_onnxruntime':reference['onnxruntime'],'onnxruntime':ort.__version__,
        'providers':{'reference':reference['provider'],'sdk':session.get_providers()},'atol':1e-4,'rtol':1e-3,
        'toolchain_image_id':case['toolchain_image_id'],'producer':producer,'cases':checks,
        'scope':'Frozen source-export host CPU ONNX versus cached CPU SDK; no CUDA PTQ or board execution','board':'not-run'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'onnxruntime':ort.__version__,'reference_onnxruntime':reference['onnxruntime']}),flush=True)
    raise SystemExit(1 if failed else 0)
