"""Compare native candidate decode with canonical Python on actual E26n outputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,subprocess
import cv2
import numpy as np
from samples.vision.yoloe.evaluator.backends import create_predictor
from samples.vision.yoloe.runtime.python.yoloe import Config
from samples._shared.yoloe26_decode import decode_candidates
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
SCRATCH=ROOT.parent/'.coordination/yoloe-cpp-kernel-tensors'
SCRATCH.mkdir(exist_ok=True)
model=ROOT.parent/'.coordination/yoloe-export-final-v3-20260928/26n/yoloe_26n_seg_pf.onnx'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
image=ROOT/'samples/vision/yoloe/test_data/office_desk.jpg'
predictor=create_predictor('onnx','s100','26n',model,sha(model),Config())
prepared=predictor.pre_process(cv2.imread(str(image)))
outputs=list(predictor.forward(prepared).values())
# inspect_graph uses semantic insertion order: cls,box,mc per stride, then proto.
from samples._shared.yoloe26_decode import OUTPUT_SHAPES
assert [tuple(x.shape) for x in outputs]==list(OUTPUT_SHAPES)
for i,output in enumerate(outputs):output.tofile(SCRATCH/f'{i}.f32')
records=[]
for single in (True,False):
 argv=[str(ROOT.parent/'.coordination/yoloe-decode-probe'),str(SCRATCH),'1' if single else '0']
 start=datetime.now(timezone.utc).isoformat()
 result=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True)
 name='single' if single else 'multi'
 (OUT/f'{name}-native.json').write_text(result.stdout)
 (OUT/f'{name}-stderr.log').write_text(result.stderr)
 assert result.returncode==0,result.stderr
 native=json.loads(result.stdout)
 boxes,scores,labels,coefficients=decode_candidates(outputs,single_label=single)
 np.testing.assert_array_equal([r['label'] for r in native],labels)
 errors={}
 for field,expected in [('box',boxes),('score',scores),('coefficients',coefficients)]:
  observed=np.asarray([r[field] for r in native],dtype=np.float32)
  np.testing.assert_allclose(observed,expected,rtol=1e-6,atol=1e-6)
  errors[field]=float(np.max(np.abs(observed-expected),initial=0))
 records.append(dict(mode=name,argv=argv,cwd=str(ROOT),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rc=result.returncode,count=len(native),label_order='identical',max_abs_error=errors,criterion='rtol=atol=1e-6'))
report=dict(backend=predictor.identity,image_sha256=sha(image),tensor_sha256={str(i):sha(SCRATCH/f'{i}.f32') for i in range(10)},runs=records,board='not-run',masks='not-compared; candidate decoder only')
(OUT/'e26-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(records,indent=2))
