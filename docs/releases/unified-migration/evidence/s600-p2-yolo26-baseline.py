REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
TARGET='s600'
REFERENCE='s:ultralytics_yolo26:nash-p/yolo26n_detect_nashp_640x640_nv12.hbm'
import datetime,hashlib,json,pathlib,sys
import cv2,numpy as np
root=pathlib.Path(REMOTE)
sys.path.insert(0,str(root/'current-4be175f7104f'))
from samples._shared.assets import resolve_asset,download_asset
from samples._shared.platforms import require_execution_target
require_execution_target(TARGET)
asset=resolve_asset(REFERENCE)
path=root/'models'/pathlib.Path(asset.filename).name
observed=download_asset(asset,path)
base=root/'baseline'
sys.path.insert(0,str(base/'samples/vision/ultralytics_yolo/runtime/python'))
from yolo26_det import YOLO26Detect,YOLO26DetectConfig
from yolo_platform import resolve_platform
model=YOLO26Detect(YOLO26DetectConfig(str(path),platform=resolve_platform(TARGET)))
model.set_scheduling_params(priority=0,bpu_cores=[0])
report={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':'cd74a2b241075bb21036d8d0855d0403f8e8c963','asset_reference':REFERENCE,'url':asset.url,'publisher_sha256':asset.sha256,'local_sha256':observed,'bytes':path.stat().st_size}
for key in ('model_names','input_names','input_shapes','input_dtypes','output_names','output_shapes','output_dtypes'):
 report[key]=getattr(model.model,key,None)
print('METADATA '+json.dumps(report,default=str),flush=True)
image_path=base/'samples/vision/ultralytics_yolo/test_data/bus.jpg'
boxes,scores,ids=model.predict(cv2.imread(str(image_path)))
np.savez(root/'baseline-results/yolo26-detect.npz',boxes=boxes,scores=scores,ids=ids)
print('RESULT '+json.dumps({'input_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),'boxes':boxes.tolist(),'scores':scores.tolist(),'ids':ids.tolist()}),flush=True)
