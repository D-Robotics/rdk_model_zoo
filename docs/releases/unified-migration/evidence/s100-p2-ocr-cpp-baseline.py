REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
DIGEST='33de0f6e7d2fafd9cccfb887b9b1e9240080681d923cd7912aa72a1006f6623c'
SOURCE='cd74a2b241075bb21036d8d0855d0403f8e8c963'
import datetime,hashlib,json,pathlib,subprocess,tarfile
root=pathlib.Path(REMOTE);base=root/'ocr-cpp-baseline';base.mkdir(exist_ok=True)
archive_path=root/'p2-ocr-cpp-baseline.tar'
assert hashlib.sha256(archive_path.read_bytes()).hexdigest()==DIGEST
with tarfile.open(archive_path) as archive:
 for member in archive.getmembers():
  path=(base/member.name).resolve()
  if base.resolve() not in path.parents:raise ValueError('unsafe archive member')
 archive.extractall(base)
sample=base/'platforms/s/samples/vision/paddle_ocr';entry=sample/'runtime/cpp';build=base/'build'
report={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':SOURCE,'source_archive_sha256':DIGEST,'target':'s100'}
def run(command):
 print('COMMAND '+json.dumps(command),flush=True)
 result=subprocess.run(command,cwd=base,capture_output=True,text=True)
 print(result.stdout,flush=True);print('COMMAND_STDERR '+result.stderr,flush=True);print('COMMAND_EXIT '+str(result.returncode),flush=True)
 if result.returncode:raise SystemExit(result.returncode)
run(['cmake','-S',str(entry),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release'])
run(['cmake','--build',str(build),'--parallel','2'])
det=root/'models/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm';rec=root/'models/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm'
image=sample/'test_data/gt_2322.jpg';vocab=sample/'test_data/ppocrv6_dict.txt';font=sample/'test_data/FangSong.ttf';output=base/'result.jpg'
run([str(build/'paddle_ocr'),'--det_model_path='+str(det),'--rec_model_path='+str(rec),'--test_image='+str(image),'--label_file='+str(vocab),'--font_path='+str(font),'--img_save_path='+str(output)])
assert output.is_file() and output.stat().st_size>0
for name,path in [('detector',det),('recognizer',rec),('input',image),('vocabulary',vocab),('font',font),('output_image',output),('executable',build/'paddle_ocr')]:report[name+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
print('RESULT '+json.dumps(report),flush=True)
print('OCR_CPP_BASELINE_PASS',flush=True)
