"""Audit pinned source bytes and packaged NPZ metadata without SDK or hardware."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "platforms/s/samples/vision/diffusiondrive"
PIN = "380e1a2bf42041af54be6f34935e50197cfadff9"
records=[]
for path in sorted(SOURCE.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts:
        continue
    rel=path.relative_to(SOURCE)
    original=subprocess.check_output(['git','show',f'{PIN}:samples/vision/diffusiondrive/{rel}'],cwd=ROOT)
    assert path.read_bytes()==original, rel
    records.append({'path':str(rel),'sha256':hashlib.sha256(original).hexdigest(),'bytes':len(original)})
archives={}
for path in sorted((SOURCE/'test_data').rglob('*.npz')):
    with np.load(path,allow_pickle=False) as data:
        archives[str(path.relative_to(SOURCE))]={name:{'shape':list(data[name].shape),'dtype':str(data[name].dtype),'finite':bool(np.isfinite(data[name]).all()),'min':float(data[name].min()),'max':float(data[name].max())} for name in data.files}
# Import only pure helpers with a stub module; no runtime is constructed.
old=sys.modules.get('hbm_runtime')
sys.modules['hbm_runtime']=ModuleType('hbm_runtime')
spec=importlib.util.spec_from_file_location('source_diffusiondrive',SOURCE/'runtime/python/diffusiondrive.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
if old is None:del sys.modules['hbm_runtime']
else:sys.modules['hbm_runtime']=old
counterexamples={}
q=SimpleNamespace(scale=np.array([.25,.5],np.float32),zero_point=np.array([0],np.float32),axis=1)
try:
    module.DiffusionDrive._dequantize(np.ones((1,2,3),np.int16),q)
except ValueError as exc:
    counterexamples['per_axis_scale_scalar_zero_point']={'source':'raises','message':str(exc),'expected':'broadcast scalar zero point across two channels'}
else:raise AssertionError('Expected source scalar-zero-point reshape failure')
q=SimpleNamespace(scale=np.array([-1],np.float32),zero_point=np.array([0],np.float32),axis=0)
negative=module.DiffusionDrive._quantize(np.array([1],np.float32),q,np.dtype(np.int16))
counterexamples['negative_input_scale']={'source_result':negative.tolist(),'expected':'reject nonpositive scale'}
# A wrong-rank label map broadcasts in the original evaluator instead of failing.
reference=np.array([[[0,1],[1,0]]]); wrong_rank=reference[0]
counterexamples['broadcast_label_agreement']={'reference_shape':list(reference.shape),'candidate_shape':list(wrong_rank.shape),'source_agreement':float(np.mean(reference==wrong_rank)),'expected':'reject shape mismatch before comparison'}
result={'source_commit':PIN,'files':records,'archives':archives,'counterexamples':counterexamples,'sdk':'not-run','board':'not-run'}
print(json.dumps(result,indent=2))
