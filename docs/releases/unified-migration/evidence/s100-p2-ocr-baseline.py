REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
TARGET='s100'
SOURCE='cd74a2b241075bb21036d8d0855d0403f8e8c963'
import datetime,hashlib,json,os,pathlib,sys,tarfile
import cv2,numpy as np
root=pathlib.Path(REMOTE);base=root/'ocr-baseline';base.mkdir(exist_ok=True)
with tarfile.open(root/'ocr-baseline.tar') as archive:
 for member in archive.getmembers():
  path=(base/member.name).resolve()
  if base.resolve() not in path.parents:raise ValueError('archive path escaped')
 archive.extractall(base)
sys.path.insert(0,str(root/'python-deps'))
group='x5' if TARGET=='x5' else 's';sample='paddleocr' if TARGET=='x5' else 'paddle_ocr'
source_dir=base/f'platforms/{group}/samples/vision/{sample}'
os.chdir(source_dir/'runtime/python');sys.path.insert(0,str(source_dir/'runtime/python'));sys.path.insert(0,str(base/f'platforms/{group}'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
report={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':SOURCE,'archive_sha256':digest(root/'ocr-baseline.tar'),'target':TARGET}
if TARGET=='x5':
 from paddleocr import PaddleOCR,PaddleOCRConfig
 det_path=root/'models/en_PP-OCRv3_det_640x640_nv12.bin';rec_path=root/'models/en_PP-OCRv3_rec_48x320_rgb.bin'
 image_path=source_dir/'test_data/paddleocr_test.jpg';image=cv2.imread(str(image_path))
 model=PaddleOCR(PaddleOCRConfig(det_model_path=str(det_path),rec_model_path=str(rec_path)))
 model.set_scheduling_params(priority=0,bpu_cores=[0])
 boxes,texts=model.predict(image)
else:
 from paddle_ocr import PaddleOCRDet,PaddleOCRDetConfig,PaddleOCRRec,PaddleOCRRecConfig
 det_path=root/'models/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm';rec_path=root/'models/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm'
 image_path=source_dir/'test_data/gt_2322.jpg';image=cv2.imread(str(image_path))
 vocab_path=source_dir/'test_data/ppocrv6_dict.txt'
 with vocab_path.open(encoding='utf-8') as handle:characters=['blank']+[line.rstrip('\n') for line in handle]+[' ']
 report['vocabulary_sha256']=digest(vocab_path);report['vocabulary_size']=len(characters)
 det=PaddleOCRDet(PaddleOCRDetConfig(model_path=str(det_path)));rec=PaddleOCRRec(PaddleOCRRecConfig(model_path=str(rec_path)))
 det.set_scheduling_params(priority=0,bpu_cores=[0]);rec.set_scheduling_params(priority=0,bpu_cores=[0])
 _,crops,boxes=det.predict(image)
 texts=[rec.predict(crop,characters) for crop in crops]
report.update({'input_sha256':digest(image_path),'detection_artifact_sha256':digest(det_path),'recognition_artifact_sha256':digest(rec_path),'boxes':[np.asarray(box).tolist() for box in boxes],'texts':texts})
output=root/'baseline-results/ocr.json';output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('RESULT '+json.dumps(report,ensure_ascii=False),flush=True)
