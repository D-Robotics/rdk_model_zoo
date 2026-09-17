REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
TARGET='s100'
ASSETS=[{'sample': 'resnet18', 'filename': 's100/resnet18_224x224_nv12.hbm', 'url': 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet18_224x224_nv12.hbm', 'expected_sha256': None}, {'sample': 'ultralytics_yolo', 'filename': 'nash-e/yolov8n_detect_nashe_640x640_nv12.hbm', 'url': 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.7.0/nash-e/yolov8n_detect_nashe_640x640_nv12.hbm', 'expected_sha256': None}]
DIGEST='77c9532feccd6f157b47b7df342a6a3bfff648fbcae65300feaf89729ed982cd'
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
from yolo26_det import YOLO26Detect,YOLO26DetectConfig
suffix={'x5':'bayese','s100':'nashe','s100p':'nashm','s600':'nashp'}[TARGET]
extension='bin' if TARGET=='x5' else 'hbm'
model26_path=root/'models'/f'yolo26n_detect_{suffix}_640x640_nv12.{extension}'
model26=YOLO26Detect(YOLO26DetectConfig(str(model26_path),platform=resolve_platform(TARGET),nms_thres=0.45))
model26.set_scheduling_params(priority=0,bpu_cores=[0])
boxes26,scores26,ids26=model26.predict(image)
old26=np.load(root/'baseline-results/yolo26-detect.npz')
np.testing.assert_array_equal(ids26,old26['ids'])
np.testing.assert_allclose(scores26,old26['scores'],rtol=1e-5,atol=1e-6)
np.testing.assert_allclose(boxes26,old26['boxes'],rtol=0,atol=1e-3)
print(json.dumps({'task':'yolo26-ltrb','status':'pass','artifact_sha256':hashlib.sha256(model26_path.read_bytes()).hexdigest(),'max_box_abs_diff':float(np.max(np.abs(boxes26-old26['boxes']))),'max_score_abs_diff':float(np.max(np.abs(scores26-old26['scores']))),'boxes':boxes26.tolist(),'scores':scores26.tolist(),'ids':ids26.tolist()}),flush=True)
keep26=[array.copy() for array in (boxes26,scores26,ids26)]
model26.predict(np.zeros_like(image))
for actual,expected in zip((boxes26,scores26,ids26),keep26):np.testing.assert_array_equal(actual,expected)
result26=subprocess.run(['bash',str(entry/'run.sh'),'detect','--family','yolo26','--model-path',str(model26_path),'--test-img',str(image_path),'--nms-thres','0.45','--img-save-path',str(base/'yolo26-result.jpg')],cwd='/tmp',capture_output=True,text=True)
print('YOLO26_NATIVE_STDOUT',result26.stdout,'YOLO26_NATIVE_STDERR',result26.stderr,flush=True)
assert result26.returncode==0,result26.returncode
assert (base/'yolo26-result.jpg').is_file()
print('YOLO26_RESULT_OWNERSHIP_AND_NATIVE_PASS',flush=True)
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
 # The old public wrapper delegates to canonical behavior with its own return ABI.
 import importlib.util
 relative='platforms/x5/samples/vision/resnet/runtime/python/resnet.py' if TARGET=='x5' else 'platforms/s/samples/vision/resnet18/runtime/python/resnet18.py'
 spec=importlib.util.spec_from_file_location('resnet_compatibility_check',base/relative)
 compat=importlib.util.module_from_spec(spec);sys.modules[spec.name]=compat;spec.loader.exec_module(compat)
 if TARGET=='x5':legacy=compat.ResNet(compat.ResNetConfig(model_path=str(model_path)))
 else:legacy=compat.Resnet18(compat.Resnet18Config(model_path=str(model_path)))
 legacy.set_scheduling_params(priority=0,bpu_cores=[0])
 legacy_result=legacy.predict(cv2.imread(str(image_path)),topk=5)
 if TARGET=='x5':legacy_ids,legacy_scores,_=legacy_result
 else:legacy_ids=np.array([row[0] for row in legacy_result]);legacy_scores=np.array([row[1] for row in legacy_result])
 np.testing.assert_array_equal(legacy_ids,old['ids'])
 np.testing.assert_allclose(legacy_scores,old['probs'],rtol=1e-5,atol=1e-6)
 print('LEGACY_RESNET_MATCH',flush=True)
 print('CLASSIFICATION_RESULT_OWNERSHIP_PASS',flush=True)
 print(json.dumps({'task':'classification','status':'pass','input_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),'artifact_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'max_score_abs_diff':float(np.max(np.abs(result.scores-old['probs']))),'ids':result.class_ids.tolist(),'scores':result.scores.tolist()}),flush=True)
 entry=base/'samples/vision/resnet/runtime/python'
 result=subprocess.run(['bash',str(entry/'run.sh'),'--asset-id',reference,'--model-path',str(model_path),'--img-save-path',str(base/'classification-result.jpg')],cwd='/tmp',capture_output=True,text=True)
 print('RESNET_NATIVE_STDOUT',result.stdout,'RESNET_NATIVE_STDERR',result.stderr,flush=True)
 assert result.returncode==0,result.returncode
 assert (base/'classification-result.jpg').is_file()
else:print('CLASSIFICATION_NOT_RUN: no published S100P ResNet18 asset',flush=True)
print('P2_PROTOCOL_COMPARISON_PASS',flush=True)
