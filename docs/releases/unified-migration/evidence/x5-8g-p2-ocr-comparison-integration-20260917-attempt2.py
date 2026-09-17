REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
TARGET='x5'
DIGEST='7e75debcc4cc212b4a36560b1488cd45b111024b7a86cb9202d5238b50232e3f'
import datetime,hashlib,json,os,pathlib,py_compile,subprocess,sys,tarfile
import cv2,numpy as np
root=pathlib.Path(REMOTE);archive_path=root/'p2-ocr-current.tar'
assert hashlib.sha256(archive_path.read_bytes()).hexdigest()==DIGEST
base=root/('ocr-current-'+DIGEST[:12]);base.mkdir(exist_ok=True)
with tarfile.open(archive_path) as archive:
 for member in archive.getmembers():
  path=(base/member.name).resolve()
  if base.resolve() not in path.parents or not member.isfile():raise ValueError('unsafe archive member')
 archive.extractall(base)
sys.path.insert(0,str(root/'python-deps'));sys.path.insert(0,str(base))
from samples._shared.platforms import detect_target
assert detect_target()==TARGET,(detect_target(),TARGET)
for path in (base/'samples').rglob('*.py'):py_compile.compile(str(path),doraise=True)
from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
from samples.vision.paddle_ocr.runtime.python.model_runner import create_stage_runners
from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline
if TARGET=='x5':
 det_name='en_PP-OCRv3_det_640x640_nv12.bin';rec_name='en_PP-OCRv3_rec_48x320_rgb.bin'
 prefix='x5:paddleocr:'
else:
 det_name='PP-OCRv6_det_infer-deploy_640x640_nv12.hbm';rec_name='PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm'
 prefix='s:paddle_ocr:s100/'
det_path=root/'models'/det_name;rec_path=root/'models'/rec_name
pair=resolve_pair(TARGET,det_asset_id=prefix+det_name,rec_asset_id=prefix+rec_name,det_model_path=str(det_path),rec_model_path=str(rec_path))
detector,recognizer=create_stage_runners(pair,priority=0,bpu_cores=[0])
pipeline=OCRPipeline(pair,detector,recognizer)
print('CONTEXT '+json.dumps({'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target':TARGET,'archive_sha256':DIGEST,'detector_sha256':hashlib.sha256(det_path.read_bytes()).hexdigest(),'recognizer_sha256':hashlib.sha256(rec_path.read_bytes()).hexdigest(),'detector_asset':prefix+det_name,'recognizer_asset':prefix+rec_name}),flush=True)
env=dict(os.environ);env['PYTHONPATH']=str(root/'python-deps')+os.pathsep+str(base)
entry=base/'samples/vision/paddle_ocr/runtime/python/main.py'
for variant in ('default','aspect'):
 golden=np.load(root/f'baseline-results/ocr-stages-{variant}.npz',allow_pickle=False)
 expected=json.loads((root/f'baseline-results/ocr-stages-{variant}.json').read_text(encoding='utf-8'))
 image=golden['image'];differences=[]
 det_inputs=pipeline.prepare_detection(image)
 expected_names={key[len('det_input_'):] for key in golden.files if key.startswith('det_input_')}
 assert set(det_inputs)==expected_names,(set(det_inputs),expected_names)
 for name,value in det_inputs.items():
  prior=golden['det_input_'+name];assert value.dtype==prior.dtype and value.shape==prior.shape
  np.testing.assert_array_equal(value,prior)
 det_outputs=detector(det_inputs)
 assert set(det_outputs)=={key[len('det_output_'):] for key in golden.files if key.startswith('det_output_')}
 for name,value in det_outputs.items():
  prior=golden['det_output_'+name];assert value.dtype==prior.dtype and value.shape==prior.shape
  np.testing.assert_allclose(value,prior,atol=1e-6,rtol=1e-5)
  differences.append(float(np.max(np.abs(value-prior))))
 detection=pipeline.postprocess_detection(det_outputs,image)
 np.testing.assert_array_equal(np.asarray(detection.boxes),golden['boxes'])
 assert len(detection.crops)==len(expected['texts'])
 texts=[]
 for index,crop in enumerate(detection.crops):
  assert crop.dtype==golden[f'crop_{index}'].dtype and crop.shape==golden[f'crop_{index}'].shape
  np.testing.assert_array_equal(crop,golden[f'crop_{index}'])
  inputs=pipeline.prepare_recognition(crop);assert len(inputs)==1
  prepared=next(iter(inputs.values()));prior_input=golden[f'rec_input_{index}']
  assert prepared.dtype==prior_input.dtype and prepared.shape==prior_input.shape
  np.testing.assert_array_equal(prepared,prior_input)
  outputs=recognizer(inputs);assert len(outputs)==1
  value=next(iter(outputs.values()));prior=golden[f'rec_output_{index}']
  assert value.dtype==prior.dtype and value.shape==prior.shape
  np.testing.assert_allclose(value,prior,atol=1e-6,rtol=1e-5)
  differences.append(float(np.max(np.abs(value-prior))))
  texts.append(pipeline.decode_recognition(outputs))
 assert texts==expected['texts'],(texts,expected['texts'])
 result=pipeline.predict(image)
 np.testing.assert_array_equal(np.asarray(result.boxes),golden['boxes']);assert list(result.texts)==expected['texts']
 saved_boxes=np.asarray(result.boxes).copy();saved_texts=list(result.texts)
 pipeline.predict(np.zeros((113,257,3),dtype=np.uint8))
 np.testing.assert_array_equal(np.asarray(result.boxes),saved_boxes);assert list(result.texts)==saved_texts
 # Lossless fixture copy makes the CLI use the exact decoded baseline pixels.
 image_path=base/f'ocr-{variant}.png';assert cv2.imwrite(str(image_path),image)
 output_path=base/f'ocr-{variant}.json'
 command=['python3',str(entry),'--target',TARGET,'--det-asset-id',prefix+det_name,'--rec-asset-id',prefix+rec_name,'--det-model-path',str(det_path),'--rec-model-path',str(rec_path),'--test-img',str(image_path),'--output-format','json','--json-output',str(output_path)]
 process=subprocess.run(command,cwd='/tmp',env=env,capture_output=True,text=True)
 print('COMMAND '+json.dumps(command),flush=True);print('NATIVE_STDOUT',process.stdout,'NATIVE_STDERR',process.stderr,flush=True)
 assert process.returncode==0,process.returncode
 native=json.loads(output_path.read_text(encoding='utf-8'))
 np.testing.assert_array_equal(native['boxes'],golden['boxes']);assert native['texts']==expected['texts']
 print('RESULT '+json.dumps({'target':TARGET,'variant':variant,'count':len(texts),'texts':texts,'input_shape':image.shape,'max_raw_output_abs_diff':max(differences,default=0.0),'input_crop_box_exact':True,'native_json_exact':True,'result_lifetime':True},ensure_ascii=False),flush=True)
wrong='s100' if TARGET=='x5' else 'x5'
command[command.index('--target')+1]=wrong
process=subprocess.run(command,cwd='/tmp',env=env,capture_output=True,text=True)
assert process.returncode!=0,'mismatched target accepted'
print('WRONG_TARGET_REJECTED '+process.stderr,flush=True)
# Exercise old public wrappers using the same fixed baseline pixels.
import importlib.util
relative='platforms/x5/samples/vision/paddleocr/runtime/python/paddleocr.py' if TARGET=='x5' else 'platforms/s/samples/vision/paddle_ocr/runtime/python/paddle_ocr.py'
spec=importlib.util.spec_from_file_location('ocr_compatibility_check',base/relative)
legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
if TARGET=='x5':
 old=legacy.PaddleOCR(legacy.PaddleOCRConfig(det_model_path=str(det_path),rec_model_path=str(rec_path)))
 old.set_scheduling_params(priority=0,bpu_cores=[0])
else:
 olddet=legacy.PaddleOCRDet(legacy.PaddleOCRDetConfig(model_path=str(det_path)))
 oldrec=legacy.PaddleOCRRec(legacy.PaddleOCRRecConfig(model_path=str(rec_path)))
 olddet.set_scheduling_params(priority=0,bpu_cores=[0]);oldrec.set_scheduling_params(priority=0,bpu_cores=[0])
 vocab=base/'samples/vision/paddle_ocr/test_data/s100/ppocrv6_dict.txt'
 chars=['blank']+vocab.read_text(encoding='utf-8').splitlines()+[' ']
for variant in ('default','aspect'):
 fixture=np.load(root/f'baseline-results/ocr-stages-{variant}.npz')
 expected=json.loads((root/f'baseline-results/ocr-stages-{variant}.json').read_text())
 if TARGET=='x5':boxes,texts=old.predict(fixture['image'])
 else:
  _,crops,boxes=olddet.predict(fixture['image']);texts=[oldrec.predict(crop,chars) for crop in crops]
 np.testing.assert_array_equal(boxes,fixture['boxes']);assert texts==expected['texts'],(texts,expected['texts'])
 print('LEGACY_OCR_EXACT '+variant,flush=True)
print('OCR_STAGE_COMPARISON_PASS',flush=True)
