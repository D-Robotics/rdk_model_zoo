REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
ASSETS=[{'sample': 'resnet18', 'filename': 's100/resnet18_224x224_nv12.hbm', 'url': 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet18_224x224_nv12.hbm', 'expected_sha256': None}, {'sample': 'ultralytics_yolo', 'filename': 'nash-e/yolov8n_detect_nashe_640x640_nv12.hbm', 'url': 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.7.0/nash-e/yolov8n_detect_nashe_640x640_nv12.hbm', 'expected_sha256': None}]
import hashlib,json,pathlib,platform,urllib.request,importlib.metadata,datetime
root=pathlib.Path(REMOTE)/'models'
root.mkdir(parents=True,exist_ok=True)
print(json.dumps({'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':platform.python_version(),'runtime_distribution':importlib.metadata.version('hbm-runtime')}),flush=True)
import hbm_runtime
for asset in ASSETS:
 p=root/pathlib.Path(asset['filename']).name
 if not p.exists():
  partial=p.with_suffix(p.suffix+'.part')
  urllib.request.urlretrieve(asset['url'],partial)
  if not partial.stat().st_size:raise ValueError('empty model')
  partial.rename(p)
 digest=hashlib.sha256(p.read_bytes()).hexdigest()
 if asset['expected_sha256'] and digest!=asset['expected_sha256']:raise ValueError('artifact digest mismatch')
 m=hbm_runtime.HB_HBMRuntime(str(p))
 report=dict(asset,local_path=str(p),local_sha256=digest,bytes=p.stat().st_size)
 for name in ['model_names','input_names','input_shapes','input_dtypes','output_names','output_shapes','output_dtypes']:
  report[name]=getattr(m,name,None)
 print('METADATA '+json.dumps(report,default=str),flush=True)
 del m
