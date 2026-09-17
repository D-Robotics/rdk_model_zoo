REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
TARGET='s100'
import datetime,hashlib,importlib.util,json,pathlib,sys
root=pathlib.Path(REMOTE)
sys.path.insert(0,str(root/'current-c0920d954fb1'))
from samples._shared.assets import list_assets,download_asset
from samples._shared.platforms import require_execution_target
require_execution_target(TARGET)
print('ENV '+json.dumps({name:importlib.util.find_spec(name) is not None for name in ('hbm_runtime','pyclipper','cv2','numpy','PIL')}),flush=True)
import hbm_runtime
group='x5' if TARGET=='x5' else 's'
sample='paddleocr' if TARGET=='x5' else 'paddle_ocr'
for asset in list_assets(group,sample):
 path=root/'models'/pathlib.Path(asset.filename).name
 observed=download_asset(asset,path)
 model=hbm_runtime.HB_HBMRuntime(str(path))
 report={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target':TARGET,'asset_reference':asset.reference,'publisher_sha256':asset.sha256,'url':asset.url,'local_sha256':observed,'bytes':path.stat().st_size}
 for key in ('model_names','input_names','input_shapes','input_dtypes','output_names','output_shapes','output_dtypes'):
  report[key]=getattr(model,key,None)
 print('METADATA '+json.dumps(report,default=str),flush=True)
 del model
