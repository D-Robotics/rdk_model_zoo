REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
TARGET='x5'
ASSETS=[{'sample': 'resnet', 'filename': 'resnet18_224x224_nv12.bin', 'url': 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/resnet18_224x224_nv12.bin', 'expected_sha256': None}, {'sample': 'ultralytics_yolo', 'filename': 'yolov8n_detect_bayese_640x640_nv12.bin', 'url': 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ultralytics_YOLO/yolov8n_detect_bayese_640x640_nv12.bin', 'expected_sha256': None}]
DIGEST='c0920d954fb1065aa1e298dce37bdfb94ada0c541a5907eaaf0a5376545b8b43'
import datetime,hashlib,json,pathlib,sys,tarfile,subprocess,py_compile,importlib.util
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
saved=[array.copy() for array in (boxes,scores,ids)]
model.predict(np.zeros_like(image))
for actual,expected in zip((boxes,scores,ids),saved):np.testing.assert_array_equal(actual,expected)
legacy_file=('platforms/x5/samples/vision/ultralytics_yolo/runtime/python/ultralytics_yolo_det.py' if TARGET=='x5' else 'platforms/s/samples/vision/ultralytics_yolo/runtime/python/yolo_detect.py')
spec=importlib.util.spec_from_file_location('legacy_detect_pilot',base/legacy_file)
legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
legacy_name='UltralyticsYOLODetect' if TARGET=='x5' else 'YoloDetect'
legacy_model=getattr(legacy,legacy_name)(getattr(legacy,legacy_name+'Config')(str(model_path)))
legacy_model.set_scheduling_params(priority=0,bpu_cores=[0])
for actual,expected in zip(legacy_model.predict(image),(boxes,scores,ids)):np.testing.assert_array_equal(actual,expected)
print('DETECT_RESULT_OWNERSHIP_AND_LEGACY_PASS',flush=True)
print(json.dumps({'task':'detect','status':'pass','input_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),'artifact_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'max_box_abs_diff':float(np.max(np.abs(boxes-old['boxes']))),'max_score_abs_diff':float(np.max(np.abs(scores-old['scores']))),'boxes':boxes.tolist(),'scores':scores.tolist(),'ids':ids.tolist()}),flush=True)
entry=base/'samples/vision/ultralytics_yolo/runtime/python'
result=subprocess.run(['bash',str(entry/'run.sh'),'detect','--family','yolov8','--model-path',str(model_path),'--test-img',str(image_path),'--img-save-path',str(base/'detect-result.jpg')],cwd='/tmp',capture_output=True,text=True)
print('YOLO_NATIVE_STDOUT',result.stdout,'YOLO_NATIVE_STDERR',result.stderr,flush=True)
assert result.returncode==0,result.returncode
assert (base/'detect-result.jpg').is_file()
wrong='s600' if TARGET!='s600' else 'x5'
result=subprocess.run(['python3',str(entry/'main.py'),'--target',wrong,'--model-path',str(model_path)],cwd='/tmp',capture_output=True,text=True)
assert result.returncode==2 and 'Target mismatch' in result.stderr,(result.returncode,result.stdout,result.stderr)
print('DETECT_MISMATCH_REJECTED_BEFORE_RUNTIME',flush=True)
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
 saved_ids=result.class_ids.copy();saved_scores=result.scores.copy()
 ClassificationTask(runner,binding).predict(np.zeros_like(cv2.imread(str(image_path))))
 np.testing.assert_array_equal(result.class_ids,saved_ids);np.testing.assert_array_equal(result.scores,saved_scores)
 print('CLASSIFICATION_RESULT_OWNERSHIP_PASS',flush=True)
 print(json.dumps({'task':'classification','status':'pass','input_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),'artifact_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'max_score_abs_diff':float(np.max(np.abs(result.scores-old['probs']))),'ids':result.class_ids.tolist(),'scores':result.scores.tolist()}),flush=True)
 entry=base/'samples/vision/resnet/runtime/python'
 result=subprocess.run(['bash',str(entry/'run.sh'),'--asset-id',reference,'--model-path',str(model_path),'--img-save-path',str(base/'classification-result.jpg')],cwd='/tmp',capture_output=True,text=True)
 print('RESNET_NATIVE_STDOUT',result.stdout,'RESNET_NATIVE_STDERR',result.stderr,flush=True)
 assert result.returncode==0,result.returncode
 assert (base/'classification-result.jpg').is_file()
else:print('CLASSIFICATION_NOT_RUN: no published S100P ResNet18 asset',flush=True)
print('PILOT_COMPARISON_PASS',flush=True)
