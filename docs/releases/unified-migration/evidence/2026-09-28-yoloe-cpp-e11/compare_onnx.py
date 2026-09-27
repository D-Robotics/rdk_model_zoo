"""Actual E11 ONNX candidate comparison; no SDK, masks or board claims."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
import cv2
import numpy as np
from samples.vision.yoloe.evaluator.backends import create_predictor
from samples.vision.yoloe.runtime.python.yoloe import Config
from samples.vision.ultralytics_yolo.runtime.python.rdk_yolo_utils import postprocess as post
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
image=ROOT/'samples/vision/yoloe/test_data/office_desk.jpg'
records=[]
for variant in ('11s','11m','11l'):
 scratch=ROOT.parent/'.coordination/yoloe-cpp-e11-tensors'/variant;scratch.mkdir(parents=True,exist_ok=True)
 model=ROOT.parent/f'.coordination/yoloe-export-final-v3-20260928/{variant}/yoloe_{variant}_seg_pf.onnx'
 predictor=create_predictor('onnx','x5',variant,model,sha(model),Config())
 prepared=predictor.pre_process(cv2.imread(str(image)))
 heads=predictor.forward(prepared)
 for i,output in enumerate(heads.values()):output.tofile(scratch/f'{i}.f32')
 boxes=[];scores=[];labels=[];coefficients=[]
 for stride in (8,16,32):
  conf,cls,selected=post.filter_classification(heads[f'cls_{stride}'],-np.log(1/0.25-1))
  boxes.append(post.decode_boxes(heads[f'box_{stride}'],selected,640//stride,stride,np.arange(16,dtype=np.float32)[None,None,:]))
  scores.append(conf);labels.append(cls);coefficients.append(post.filter_mces(heads[f'mces_{stride}'],selected))
 boxes,scores,labels,coefficients=[np.concatenate(x) for x in (boxes,scores,labels,coefficients)]
 keep=post.NMS(boxes,scores,labels,0.7)
 boxes,scores,labels,coefficients=[x[keep] for x in (boxes,scores,labels,coefficients)]
 (OUT/f'{variant}-python.json').write_text(json.dumps([dict(label=int(label),score=float(score),box=box.tolist(),coefficients=coefficient.tolist()) for label,score,box,coefficient in zip(labels,scores,boxes,coefficients)],indent=2)+'\n')
 argv=[str(ROOT.parent/'.coordination/yoloe-decode-probe'),str(scratch),'1','11']
 start=datetime.now(timezone.utc).isoformat();result=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True)
 (OUT/f'{variant}-native.json').write_text(result.stdout);(OUT/f'{variant}-stderr.log').write_text(result.stderr)
 assert result.returncode==0,result.stderr
 native=json.loads(result.stdout)
 np.testing.assert_array_equal([r['label'] for r in native],labels)
 errors={}
 for field,expected in [('box',boxes),('score',scores),('coefficients',coefficients)]:
  observed=np.asarray([r[field] for r in native],dtype=np.float32)
  rtol,atol=(1e-5,1e-4) if field=='box' else (1e-6,1e-6)
  np.testing.assert_allclose(observed,expected,rtol=rtol,atol=atol)
  errors[field]=dict(max_abs_error=float(np.max(np.abs(observed-expected),initial=0)),rtol=rtol,atol=atol)
 records.append(dict(variant=variant,model=predictor.identity,image_sha256=sha(image),tensor_sha256={str(i):sha(scratch/f'{i}.f32') for i in range(10)},argv=argv,cwd=str(ROOT),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rc=result.returncode,count=len(native),label_order='identical',numerics=errors))
 (OUT/'comparison.json').write_text(json.dumps(dict(runs=records,board='not-run',masks='not-compared; candidates only',boundary_difference='Native suppresses IoU > threshold; Python suppresses IoU >= threshold. Exact-score ties also have distinct documented ordering.'),indent=2)+'\n')
 print(variant,len(native),errors,flush=True)
