REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
DIGEST='ba1d74010890f326173ace7b5aa4700b1a629380cb629aad6c224f503424e99e'
import pathlib,subprocess,os,json
root=pathlib.Path(REMOTE);base=root/('integration-cpp-'+DIGEST[:12])
for name,binary in [('resnet','resnet18'),('paddle_ocr','paddle_ocr')]:
 for mode in ['canonical','compatibility']:
  entry=base/'samples/vision'/name/'runtime/cpp' if mode=='canonical' else base/'platforms/s/samples/vision'/('resnet18' if name=='resnet' else name)/'runtime/cpp'
  env=dict(os.environ,BUILD_DIR=str(base/(name+'-canonical')),JOBS='2',BUILD_JOBS='2')
  if name=='resnet':
   args=['--model_path='+str(root/'models/resnet18_224x224_nv12.hbm'),'--test_img='+str(root/'resnet-cpp-baseline/platforms/s/samples/vision/resnet18/test_data/zebra_cls.jpg'),'--label_file='+str(base/'platforms/s/datasets/imagenet/imagenet_classes.names')]
  else:
   data=root/'ocr-cpp-baseline/platforms/s/samples/vision/paddle_ocr/test_data'
   args=['--det_model_path='+str(root/'models/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm'),'--rec_model_path='+str(root/'models/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm'),'--test_image='+str(data/'gt_2322.jpg'),'--label_file='+str(data/'ppocrv6_dict.txt'),'--font_path='+str(data/'FangSong.ttf'),'--img_save_path='+str(base/('launcher-'+mode+'.jpg'))]
  command=['bash',str(entry/'run.sh')]+args
  result=subprocess.run(command,cwd='/tmp',env=env,capture_output=True,text=True)
  print(json.dumps({'sample':name,'entry':mode,'exit':result.returncode}),flush=True)
  print(result.stdout,flush=True);print(result.stderr,flush=True)
  assert result.returncode==0
  if name=='resnet':
   assert [x for x in result.stdout.splitlines() if x.startswith('TOP-')]==json.loads((root/'baseline-results/resnet-cpp.json').read_text())
  else:
   import numpy as np,cv2
   np.testing.assert_array_equal(cv2.imread(str(base/('launcher-'+mode+'.jpg'))),cv2.imread(str(root/'ocr-cpp-baseline/result.jpg')))
print('CPP_LAUNCHERS_PASS',flush=True)
