REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
TARGET='x5'
import hashlib,json,os,pathlib,sys
import cv2,numpy as np
root=pathlib.Path(REMOTE);base=root/'ocr-baseline'
sys.path.insert(0,str(root/'python-deps'))
group='x5' if TARGET=='x5' else 's';sample='paddleocr' if TARGET=='x5' else 'paddle_ocr'
source_dir=base/f'platforms/{group}/samples/vision/{sample}'
os.chdir(source_dir/'runtime/python');sys.path.insert(0,str(source_dir/'runtime/python'));sys.path.insert(0,str(base/f'platforms/{group}'))
if TARGET=='x5':
 from paddleocr import PaddleOCR,PaddleOCRConfig
 model=PaddleOCR(PaddleOCRConfig(det_model_path=str(root/'models/en_PP-OCRv3_det_640x640_nv12.bin'),rec_model_path=str(root/'models/en_PP-OCRv3_rec_48x320_rgb.bin')))
 model.set_scheduling_params(priority=0,bpu_cores=[0])
 source_image=cv2.imread(str(source_dir/'test_data/paddleocr_test.jpg'))
else:
 from paddle_ocr import PaddleOCRDet,PaddleOCRDetConfig,PaddleOCRRec,PaddleOCRRecConfig
 det=PaddleOCRDet(PaddleOCRDetConfig(model_path=str(root/'models/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm')))
 rec=PaddleOCRRec(PaddleOCRRecConfig(model_path=str(root/'models/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm')))
 det.set_scheduling_params(priority=0,bpu_cores=[0]);rec.set_scheduling_params(priority=0,bpu_cores=[0])
 source_image=cv2.imread(str(source_dir/'test_data/gt_2322.jpg'))
 with (source_dir/'test_data/ppocrv6_dict.txt').open(encoding='utf-8') as handle:characters=['blank']+[line.rstrip('\n') for line in handle]+[' ']
def flat(nested):return next(iter(nested.values()))
for variant,image in [('default',source_image),('aspect',cv2.resize(source_image,(713,509),interpolation=cv2.INTER_LINEAR))]:
 arrays={'image':image};texts=[]
 if TARGET=='x5':
  inputs=model.pre_process(image);outputs=model.forward(inputs)
  for key,value in flat(inputs).items():arrays['det_input_'+key]=value.copy()
  for key,value in flat(outputs).items():arrays['det_output_'+key]=value.copy()
  _,boxes=model.post_process(outputs,image)
  crops=[model._crop_and_rotate(image,box) for box in boxes]
 else:
  inputs=det.pre_process(image);outputs=det.forward(inputs)
  for key,value in flat(inputs).items():arrays['det_input_'+key]=value.copy()
  for key,value in flat(outputs).items():arrays['det_output_'+key]=value.copy()
  _,crops,boxes=det.post_process(outputs,image,image.shape[1],image.shape[0])
 arrays['boxes']=np.asarray(boxes)
 for index,crop in enumerate(crops):
  arrays[f'crop_{index}']=crop.copy()
  if TARGET=='x5':
   inputs=model._rec_pre_process(crop);outputs=model._rec_forward(inputs)
   arrays[f'rec_input_{index}']=inputs.copy()
   _,text=model._rec_post_process(outputs)
  else:
   inputs=rec.pre_process(crop);outputs=rec.forward(inputs)
   arrays[f'rec_input_{index}']=next(iter(flat(inputs).values())).copy()
   text=rec.post_process(outputs,characters)
  arrays[f'rec_output_{index}']=next(iter(flat(outputs).values())).copy();texts.append(text)
 output=root/f'baseline-results/ocr-stages-{variant}.npz';np.savez_compressed(output,**arrays)
 report={'target':TARGET,'variant':variant,'image_shape':image.shape,'boxes':arrays['boxes'].tolist(),'texts':texts,'stage_archive_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'arrays':{key:{'shape':value.shape,'dtype':str(value.dtype),'sha256':hashlib.sha256(value.tobytes()).hexdigest()} for key,value in arrays.items()}}
 (root/f'baseline-results/ocr-stages-{variant}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print('RESULT '+json.dumps(report,ensure_ascii=False),flush=True)
print('OCR_STAGES_BASELINE_PASS',flush=True)
