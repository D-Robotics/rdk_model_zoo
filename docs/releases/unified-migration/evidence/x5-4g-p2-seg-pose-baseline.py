REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
TARGET='x5'
REFERENCES={'seg': 'x5:ultralytics_yolo:yolov8n_seg_bayese_640x640_nv12.bin', 'pose': 'x5:ultralytics_yolo:yolov8n_pose_bayese_640x640_nv12.bin'}
import datetime,hashlib,json,pathlib,sys
import cv2,numpy as np
root=pathlib.Path(REMOTE)
sys.path.insert(0,str(root/'current-c0920d954fb1'))
from samples._shared.assets import resolve_asset,download_asset
from samples._shared.platforms import require_execution_target
require_execution_target(TARGET)
base=root/'baseline';sys.path.insert(0,str(base/'samples/vision/ultralytics_yolo/runtime/python'))
from yolo_seg import YoloSeg,YoloSegConfig
from yolo_pose import YoloPose,YoloPoseConfig
from yolo_platform import resolve_platform
image_path=base/'samples/vision/ultralytics_yolo/test_data/bus.jpg';image=cv2.imread(str(image_path))
for task,reference in REFERENCES.items():
 asset=resolve_asset(reference);path=root/'models'/pathlib.Path(asset.filename).name
 observed=download_asset(asset,path)
 constructor,config=(YoloSeg,YoloSegConfig) if task=='seg' else (YoloPose,YoloPoseConfig)
 model=constructor(config(str(path),platform=resolve_platform(TARGET)))
 model.set_scheduling_params(priority=0,bpu_cores=[0])
 report={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':'cd74a2b241075bb21036d8d0855d0403f8e8c963','task':task,'asset_reference':reference,'url':asset.url,'publisher_sha256':asset.sha256,'local_sha256':observed,'bytes':path.stat().st_size}
 for key in ('model_names','input_names','input_shapes','input_dtypes','output_names','output_shapes','output_dtypes'):report[key]=getattr(model.model,key,None)
 print('METADATA '+json.dumps(report,default=str),flush=True)
 result=model.predict(image);arrays=dict(zip(('boxes','scores','ids'),result[:3]))
 if task=='seg':
  arrays.update({f'mask_{index}':mask for index,mask in enumerate(result[3])})
 else:arrays.update({'keypoints':result[3],'keypoint_scores':result[4]})
 np.savez_compressed(root/f'baseline-results/yolov8-{task}.npz',**arrays)
 print('RESULT '+json.dumps({'task':task,'input_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),'boxes':result[0].tolist(),'scores':result[1].tolist(),'ids':result[2].tolist(),'arrays':{key:{'shape':value.shape,'dtype':str(value.dtype),'sha256':hashlib.sha256(value.tobytes()).hexdigest()} for key,value in arrays.items()}}),flush=True)
 del model
print('SEG_POSE_BASELINE_PASS',flush=True)
