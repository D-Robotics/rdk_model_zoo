REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
TARGET='s100'
ASSETS=[{'sample': 'resnet18', 'filename': 's100/resnet18_224x224_nv12.hbm', 'url': 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet18_224x224_nv12.hbm', 'expected_sha256': None}, {'sample': 'ultralytics_yolo', 'filename': 'nash-e/yolov8n_detect_nashe_640x640_nv12.hbm', 'url': 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.7.0/nash-e/yolov8n_detect_nashe_640x640_nv12.hbm', 'expected_sha256': None}]
DIGEST='d79826ba0dba69f4cdac4bbb42dc098c64a9ba2aa19596857ce8c38f16858ccf'
import datetime,hashlib,json,pathlib,sys,tarfile,subprocess,py_compile
import cv2,numpy as np
root=pathlib.Path(REMOTE)
assert hashlib.sha256((root/'current.tar').read_bytes()).hexdigest()==DIGEST
base=root/('current-'+DIGEST[:12]);base.mkdir(exist_ok=True)
with tarfile.open(root/'current.tar') as archive:
 for member in archive.getmembers():
  target=(base/member.name).resolve()
  if base.resolve() not in target.parents or not member.isfile():raise ValueError('unsafe archive member')
 archive.extractall(base)
sys.path.insert(0,str(base));sys.path.insert(0,str(base/'samples/vision/ultralytics_yolo/runtime/python'))
from samples._shared.platforms import detect_target
assert detect_target()==TARGET,(detect_target(),TARGET)
print(json.dumps({'archive_sha256':DIGEST,'detected_target':detect_target(),'started':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
for path in list((base/'samples/_shared').rglob('*.py'))+list((base/'samples/vision/resnet/runtime').rglob('*.py'))+list((base/'samples/vision/ultralytics_yolo/runtime/python').rglob('*.py')):
 py_compile.compile(str(path),doraise=True)
print('COMPILE_PASS',flush=True)
from yolo_detect import YoloDetect,YoloDetectConfig
from yolo_platform import resolve_platform
asset=next(a for a in ASSETS if a['sample']=='ultralytics_yolo')
model_path=root/'models'/pathlib.Path(asset['filename']).name
image_path=base/'samples/vision/ultralytics_yolo/test_data/bus.jpg'
image=cv2.imread(str(image_path))
model=YoloDetect(YoloDetectConfig(str(model_path),platform=resolve_platform(TARGET)))
model.set_scheduling_params(priority=0,bpu_cores=[0])
boxes,scores,ids=model.predict(image)
old=np.load(root/'baseline-results/detect.npz')
np.testing.assert_array_equal(ids,old['ids'])
np.testing.assert_allclose(scores,old['scores'],rtol=1e-5,atol=1e-6)
np.testing.assert_allclose(boxes,old['boxes'],rtol=0,atol=1e-3)
print(json.dumps({'task':'detect','status':'pass','input_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),'artifact_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'max_box_abs_diff':float(np.max(np.abs(boxes-old['boxes']))),'max_score_abs_diff':float(np.max(np.abs(scores-old['scores']))),'boxes':boxes.tolist(),'scores':scores.tolist(),'ids':ids.tolist()}),flush=True)
entry=base/'samples/vision/ultralytics_yolo/runtime/python'
result=subprocess.run(['bash',str(entry/'run.sh'),'detect','--family','yolov8','--model-path',str(model_path),'--test-img',str(image_path),'--img-save-path',str(base/'detect-result.jpg')],cwd='/tmp',capture_output=True,text=True)
print('YOLO_NATIVE_STDOUT',result.stdout,'YOLO_NATIVE_STDERR',result.stderr,flush=True)
assert result.returncode==0,result.returncode
assert (base/'detect-result.jpg').is_file()
if TARGET!='s100p':
 from samples.vision.resnet.runtime.python.model_binding import resolve_selection
 from samples.vision.resnet.runtime.python.model_runner import RuntimeModelRunner
 from samples.vision.resnet.runtime.python.classification import ClassificationTask
 asset=next(a for a in ASSETS if a['sample']!='ultralytics_yolo')
 model_path=root/'models'/pathlib.Path(asset['filename']).name
 reference=('x5' if TARGET=='x5' else 's')+':'+asset['sample']+':'+asset['filename']
 selection=resolve_selection('auto',asset_id=reference,model_path=model_path)
 runner=RuntimeModelRunner(selection);binding=runner.load()
 runner.set_scheduling_params(priority=0,bpu_cores=[0])
 image_path=base/'samples/vision/resnet/test_data/white_wolf.JPEG'
 result=ClassificationTask(runner,binding).predict(cv2.imread(str(image_path)))
 old=np.load(root/'baseline-results/classification.npz')
 np.testing.assert_array_equal(result.class_ids,old['ids'])
 np.testing.assert_allclose(result.scores,old['probs'],rtol=1e-5,atol=1e-6)
 print(json.dumps({'task':'classification','status':'pass','input_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),'artifact_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'max_score_abs_diff':float(np.max(np.abs(result.scores-old['probs']))),'ids':result.class_ids.tolist(),'scores':result.scores.tolist()}),flush=True)
 entry=base/'samples/vision/resnet/runtime/python'
 result=subprocess.run(['bash',str(entry/'run.sh'),'--asset-id',reference,'--model-path',str(model_path),'--img-save-path',str(base/'classification-result.jpg')],cwd='/tmp',capture_output=True,text=True)
 print('RESNET_NATIVE_STDOUT',result.stdout,'RESNET_NATIVE_STDERR',result.stderr,flush=True)
 assert result.returncode==0,result.returncode
 assert (base/'classification-result.jpg').is_file()
else:print('CLASSIFICATION_NOT_RUN: no published S100P ResNet18 asset',flush=True)
print('PILOT_COMPARISON_PASS',flush=True)
