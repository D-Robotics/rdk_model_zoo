"""Exact pixel comparison using hash-bound real E26n raw ONNX outputs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
import cv2,numpy as np
from samples._shared.yoloe26_decode import decode_candidates,restore_masks,OUTPUT_SHAPES
from samples._shared.yoloe26_geometry import letterbox
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
scratch=ROOT.parent/'.coordination/yoloe-cpp-kernel-tensors'
prior=json.loads((OUT.parent/'2026-09-28-yoloe-cpp-e11/e26-comparison.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
outputs=[]
for i,shape in enumerate(OUTPUT_SHAPES):
 p=scratch/f'{i}.f32';assert sha(p)==prior['tensor_sha256'][str(i)];outputs.append(np.fromfile(p,np.float32).reshape(shape))
image=ROOT/'samples/vision/yoloe/test_data/office_desk.jpg';assert sha(image)==prior['image_sha256']
pixels=cv2.imread(str(image));_,geometry=letterbox(pixels)
boxes,scores,labels,coefficients=decode_candidates(outputs)
np.concatenate([boxes,coefficients],axis=1).astype(np.float32).tofile(scratch/'mask-candidates.f32')
expected_boxes,expected_masks=restore_masks(boxes,coefficients,outputs[9][0],geometry)
token=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
destination=ROOT.parent/'.coordination'/('yoloe-mask-probe-'+token)
argv=[str(ROOT.parent/'.coordination/yoloe-native-cmake/mask_probe'),str(scratch),str(pixels.shape[1]),str(pixels.shape[0]),str(destination)]
start=datetime.now(timezone.utc).isoformat();result=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True)
(OUT/'native-masks.json').write_text(result.stdout);(OUT/'native-stderr.log').write_text(result.stderr)
assert result.returncode==0,result.stderr
(OUT/'execution.json').write_text(json.dumps(dict(argv=argv,cwd=str(ROOT),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rc=result.returncode),indent=2)+'\n')
rows=json.loads(result.stdout);assert len(rows)==len(expected_masks)
records=[];archive={};different=0
for i,(row,expected) in enumerate(zip(rows,expected_masks)):
 native=np.fromfile(destination/row['file'],np.uint8).reshape(row['height'],row['width'])
 assert native.shape==expected.shape,(i,native.shape,expected.shape)
 delta=int(np.count_nonzero(native!=expected));different+=delta
 archive[f'native_{i}']=native;archive[f'python_{i}']=expected
 records.append(dict(index=i,shape=list(native.shape),different_pixels=delta,native_sha256=hashlib.sha256(native.tobytes()).hexdigest(),python_sha256=hashlib.sha256(expected.tobytes()).hexdigest()))
np.savez_compressed(OUT/'mask-arrays.npz',**archive)
box_delta=float(np.max(np.abs(np.asarray([r['box'] for r in rows],np.float32)-expected_boxes),initial=0))
report=dict(argv=argv,cwd=str(ROOT),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rc=result.returncode,raw_tensor_reference=str(OUT.parent/'2026-09-28-yoloe-cpp-e11/e26-comparison.json'),image_sha256=sha(image),candidates_sha256=sha(scratch/'mask-candidates.f32'),opencv_python=cv2.__version__,count=len(rows),different_pixels=different,box_max_abs_error=box_delta,masks=records,board='not-run',criterion='exact mask shapes/pixels; boxes rtol=atol=1e-6')
(OUT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print('masks',len(rows),'different pixels',different,'box max error',box_delta,flush=True)
np.testing.assert_allclose(np.asarray([r['box'] for r in rows],np.float32),expected_boxes,rtol=1e-6,atol=1e-6)
assert different==0,'Mask pixels differ; inspect evidence before changing code or criteria.'
