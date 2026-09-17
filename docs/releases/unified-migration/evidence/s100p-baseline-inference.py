REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
TARGET='s100p'
ASSETS=[{'sample': 'ultralytics_yolo', 'filename': 'nash-m/yolov8n_detect_nashm_640x640_nv12.hbm', 'url': 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.7.0/nash-m/yolov8n_detect_nashm_640x640_nv12.hbm', 'expected_sha256': None}]
SOURCE='cd74a2b241075bb21036d8d0855d0403f8e8c963'
import datetime,hashlib,json,os,pathlib,sys,tarfile
import cv2,numpy as np
root=pathlib.Path(REMOTE)
base=root/'baseline'
base.mkdir(exist_ok=True)
with tarfile.open(root/'baseline.tar') as archive:
 for member in archive.getmembers():
  target=(base/member.name).resolve()
  if base.resolve() not in target.parents:raise ValueError('archive path outside baseline')
 archive.extractall(base)
out=root/'baseline-results'
out.mkdir(exist_ok=True)
print(json.dumps({'source_commit':SOURCE,'archive_sha256':hashlib.sha256((root/'baseline.tar').read_bytes()).hexdigest(),'started':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
sys.path.insert(0,str(base/'samples/vision/ultralytics_yolo/runtime/python'))
from yolo_detect import YoloDetect,YoloDetectConfig
from yolo_platform import resolve_platform
artifact=next(a for a in ASSETS if a['sample']=='ultralytics_yolo')
model_path=root/'models'/pathlib.Path(artifact['filename']).name
image_path=base/'samples/vision/ultralytics_yolo/test_data/bus.jpg'
image=cv2.imread(str(image_path))
model=YoloDetect(YoloDetectConfig(str(model_path),platform=resolve_platform(TARGET)))
model.set_scheduling_params(priority=0,bpu_cores=[0])
boxes,scores,ids=model.predict(image)
np.savez(out/'detect.npz',boxes=boxes,scores=scores,ids=ids)
print(json.dumps({'task':'detect','input_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),'boxes':boxes.tolist(),'scores':scores.tolist(),'ids':ids.tolist()}),flush=True)
if TARGET!='s100p':
 group='x5' if TARGET=='x5' else 's'
 sample='resnet' if TARGET=='x5' else 'resnet18'
 cwd=base/f'platforms/{group}/samples/vision/{sample}/runtime/python'
 os.chdir(cwd);sys.path.insert(0,str(cwd));sys.path.insert(0,str(base/f'platforms/{group}'))
 artifact=next(a for a in ASSETS if a['sample']!='ultralytics_yolo')
 model_path=root/'models'/pathlib.Path(artifact['filename']).name
 image_path=base/'platforms/x5/samples/vision/resnet/test_data/white_wolf.JPEG'
 image=cv2.imread(str(image_path))
 if TARGET=='x5':
  from resnet import ResNet,ResNetConfig
  net=ResNet(ResNetConfig(str(model_path)))
  net.set_scheduling_params(priority=0,bpu_cores=[0])
  ids,probs,labels=net.predict(image)
 else:
  from resnet18 import Resnet18,Resnet18Config
  net=Resnet18(Resnet18Config(str(model_path)))
  net.set_scheduling_params(priority=0,bpu_cores=[0])
  values=net.predict(image,topk=5)
  ids=np.array([v[0] for v in values]);probs=np.array([v[1] for v in values])
 np.savez(out/'classification.npz',ids=ids,probs=probs)
 print(json.dumps({'task':'classification','input_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),'ids':ids.tolist(),'probs':probs.tolist()}),flush=True)
print('FINISHED '+datetime.datetime.now(datetime.timezone.utc).isoformat())
