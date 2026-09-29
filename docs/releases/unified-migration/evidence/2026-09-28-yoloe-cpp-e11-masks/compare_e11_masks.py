"""Compare E11 mask math on identical, real native-decoded candidate inputs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
import cv2,numpy as np
from samples.vision.ultralytics_yolo.runtime.python.geometry import make_transform,inverse_boxes
from samples.vision.ultralytics_yolo.runtime.python.rdk_yolo_utils import postprocess as post
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
prior=OUT.parent/'2026-09-28-yoloe-cpp-e11'
reference=json.loads((prior/'comparison.json').read_text())['runs'][0]
assert reference['variant']=='11s'
scratch=ROOT.parent/'.coordination/yoloe-cpp-e11-tensors/11s'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(scratch/'9.f32')==reference['tensor_sha256']['9']
proto=np.fromfile(scratch/'9.f32',np.float32).reshape(160,160,32)
rows=json.loads((prior/'11s-native.json').read_text())
boxes=np.asarray([r['box'] for r in rows],np.float32)
coefficients=np.asarray([r['coefficients'] for r in rows],np.float32)
np.concatenate([boxes,coefficients],axis=1).tofile(scratch/'mask-candidates.f32')
image=ROOT/'samples/vision/yoloe/test_data/office_desk.jpg';assert sha(image)==reference['image_sha256']
pixels=cv2.imread(str(image));h,w=pixels.shape[:2]
context=make_transform((h,w),(640,640),1);left,top,right,bottom=context.padding
visible=boxes.copy();visible[:,[0,2]]=np.clip(visible[:,[0,2]],left,640-right);visible[:,[1,3]]=np.clip(visible[:,[1,3]],top,640-bottom)
raw=post.decode_masks(coefficients,visible,proto,640,640,160,160,mask_thresh=0.5)
expected_boxes=inverse_boxes(boxes,context)
records=[]
for morph in (False,True):
 expected=post.resize_masks_to_boxes(raw,expected_boxes,w,h,do_morph=morph)
 token=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
 destination=ROOT.parent/'.coordination'/('yoloe-e11-masks-'+token)
 argv=[str(ROOT.parent/'.coordination/yoloe-native-cmake/mask_probe'),str(scratch),str(w),str(h),str(destination),'11',str(int(morph))]
 start=datetime.now(timezone.utc).isoformat();result=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True)
 name='morph' if morph else 'plain'
 (OUT/f'{name}-native.json').write_text(result.stdout);(OUT/f'{name}-stderr.log').write_text(result.stderr)
 record=dict(name=name,argv=argv,cwd=str(ROOT),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rc=result.returncode)
 records.append(record);(OUT/'execution.json').write_text(json.dumps(records,indent=2)+'\n')
 assert result.returncode==0,result.stderr
 native_rows=json.loads(result.stdout);assert len(native_rows)==len(expected)
 details=[];archive={};different=0
 for i,(row,wanted) in enumerate(zip(native_rows,expected)):
  observed=np.fromfile(destination/row['file'],np.uint8).reshape(row['height'],row['width'])
  assert observed.shape==wanted.shape,(i,observed.shape,wanted.shape)
  delta=int(np.count_nonzero(observed!=wanted));different+=delta
  details.append(dict(index=i,shape=list(observed.shape),different_pixels=delta,native_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),python_sha256=hashlib.sha256(wanted.tobytes()).hexdigest()))
  archive[f'native_{i}']=observed;archive[f'python_{i}']=wanted
 np.savez_compressed(OUT/f'{name}-arrays.npz',**archive)
 observed_boxes=np.asarray([r['box'] for r in native_rows],np.float32)
 record.update(count=len(expected),different_pixels=different,box_max_abs_error=float(np.max(np.abs(observed_boxes-expected_boxes),initial=0)),masks=details)
 (OUT/'comparison.json').write_text(json.dumps(dict(runs=records,model_reference=str(prior/'comparison.json'),native_candidate_sha256=sha(prior/'11s-native.json'),candidate_input_sha256=sha(scratch/'mask-candidates.f32'),prototype_sha256=sha(scratch/'9.f32'),image_sha256=sha(image),criterion='exact shapes/pixels; boxes rtol=atol=1e-6',scope='Same native candidate boxes/coefficients supplied to both mask implementations; not a claim of full Python/C++ candidate equality.',board='not-run'),indent=2)+'\n')
 print(name,'masks',len(expected),'different',different,flush=True)
 np.testing.assert_allclose(observed_boxes,expected_boxes,rtol=1e-6,atol=1e-6)
 assert different==0,'Mask pixels differ; inspect before changing criteria.'
