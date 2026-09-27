"""Compare complete native prepared NV12 bytes against canonical Python."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
import cv2,numpy as np
from samples._shared.yoloe26_geometry import letterbox
from samples.vision.ultralytics_yolo.runtime.python.geometry import resize_with_transform
from samples.vision.ultralytics_yolo.runtime.python.rdk_yolo_utils.preprocess import bgr_to_nv12_planes
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
scratch=ROOT.parent/'.coordination/yoloe-stage-inputs';scratch.mkdir(exist_ok=True)
source=ROOT/'samples/vision/yoloe/test_data/office_desk.jpg'
images={'office':cv2.imread(str(source)),'odd':np.random.default_rng(713).integers(0,256,(333,1000,3),np.uint8)}
records=[];archive={}
for name,image in images.items():
 path=scratch/(name+'.bgr');image.tofile(path)
 for family,mode in [('11',1),('11',0),('26',1)]:
  case=f'{name}-{family}-{mode}'
  pixels,g=letterbox(image) if family=='26' else resize_with_transform(image,(640,640),mode)
  y,uv=bgr_to_nv12_planes(pixels)
  token=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f');dest=scratch/(case+'-'+token)
  argv=[str(ROOT.parent/'.coordination/yoloe-stage-library/tests/input_probe'),str(path),str(image.shape[1]),str(image.shape[0]),family,str(mode),str(dest)]
  start=datetime.now(timezone.utc).isoformat();p=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True)
  (OUT/(case+'.log')).write_text(p.stdout+p.stderr)
  row=dict(case=case,argv=argv,cwd=str(ROOT),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rc=p.returncode,bgr_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),log=case+'.log')
  records.append(row);(OUT/'input-comparison.json').write_text(json.dumps(records,indent=2)+'\n')
  assert p.returncode==0,p.stderr
  meta=json.loads(p.stdout);assert meta['resized']==list(g.resized_size);assert meta['padding']==list(g.padding)
  for plane,expected in [('y',y),('uv',uv)]:
   expected=np.ascontiguousarray(expected).reshape(-1);actual=np.fromfile(dest/(plane+'.u8'),np.uint8)
   np.testing.assert_array_equal(actual,expected)
   row[plane]=dict(bytes=actual.size,sha256=hashlib.sha256(actual.tobytes()).hexdigest(),different_bytes=0)
   archive[case+'-'+plane]=actual
  row['geometry']=meta
  (OUT/'input-comparison.json').write_text(json.dumps(records,indent=2)+'\n')
np.savez_compressed(OUT/'nv12-planes.npz',**archive)
print('Six complete NV12 plane comparisons passed; byte differences 0. No inference or SDK executed.')
