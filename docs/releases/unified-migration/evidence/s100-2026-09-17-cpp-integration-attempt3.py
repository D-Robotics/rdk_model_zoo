REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
DIGEST='ba1d74010890f326173ace7b5aa4700b1a629380cb629aad6c224f503424e99e'
import pathlib,hashlib,json,subprocess,tarfile
root=pathlib.Path(REMOTE);base=root/('integration-cpp-'+DIGEST[:12]);base.mkdir(exist_ok=True)
a=root/'integration-cpp.tar';assert hashlib.sha256(a.read_bytes()).hexdigest()==DIGEST
with tarfile.open(a) as t:
 for member in t.getmembers():
  if not member.isfile() or base.resolve() not in (base/member.name).resolve().parents:raise ValueError('unsafe member')
 t.extractall(base)
def run(cmd):
 print('COMMAND '+json.dumps(cmd),flush=True)
 r=subprocess.run(cmd,cwd=base,capture_output=True,text=True)
 print(r.stdout,flush=True);print(r.stderr,flush=True);print('COMMAND_EXIT '+str(r.returncode),flush=True)
 assert r.returncode==0,r.returncode
 return r.stdout
for name,binary in [('resnet','resnet18'),('paddle_ocr','paddle_ocr')]:
 sample=base/'samples/vision'/name
 for mode,entry in [('canonical',sample/'runtime/cpp'),('compatibility',base/'platforms/s/samples/vision'/('resnet18' if name=='resnet' else name)/'runtime/cpp')]:
  build=base/(name+'-'+mode)
  run(['cmake','-S',str(entry),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release'])
  run(['cmake','--build',str(build),'--parallel','2'])
  candidates=list(build.rglob(binary));assert len(candidates)==1,candidates
  exe=candidates[0]
  if name=='resnet':
   image=root/'resnet-cpp-baseline/platforms/s/samples/vision/resnet18/test_data/zebra_cls.jpg'
   labels=base/'platforms/s/datasets/imagenet/imagenet_classes.names'
   out=run([str(exe),'--model_path='+str(root/'models/resnet18_224x224_nv12.hbm'),'--test_img='+str(image),'--label_file='+str(labels)])
   rows=[line for line in out.splitlines() if line.startswith('TOP-')]
   assert rows==json.loads((root/'baseline-results/resnet-cpp.json').read_text()),rows
  else:
   original=root/'ocr-cpp-baseline/platforms/s/samples/vision/paddle_ocr/test_data'
   output=base/('ocr-'+mode+'.jpg')
   out=run([str(exe),'--det_model_path='+str(root/'models/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm'),'--rec_model_path='+str(root/'models/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm'),'--test_image='+str(original/'gt_2322.jpg'),'--label_file='+str(original/'ppocrv6_dict.txt'),'--font_path='+str(original/'FangSong.ttf'),'--img_save_path='+str(output)])
   assert output.is_file()
   import cv2,numpy as np
   np.testing.assert_array_equal(cv2.imread(str(output)),cv2.imread(str(root/'ocr-cpp-baseline/result.jpg')))
  print('CPP_RESULT '+json.dumps({'sample':name,'entry':mode,'baseline_exact':True}),flush=True)
print('INTEGRATION_CPP_PASS',flush=True)
