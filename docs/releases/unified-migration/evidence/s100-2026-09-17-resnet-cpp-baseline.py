REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
DIGEST='95f52e301dc13bac3e5c32c82bf062716944879defbe486843bc27d12a2fdab0'
SOURCE='cd74a2b241075bb21036d8d0855d0403f8e8c963'
import datetime,hashlib,json,pathlib,subprocess,tarfile
root=pathlib.Path(REMOTE);base=root/'resnet-cpp-baseline';base.mkdir(exist_ok=True)
archive_path=root/'2026-09-17-resnet-cpp-baseline.tar'
assert hashlib.sha256(archive_path.read_bytes()).hexdigest()==DIGEST
with tarfile.open(archive_path) as archive:
 for member in archive.getmembers():
  path=(base/member.name).resolve()
  if base.resolve() not in path.parents:raise ValueError('unsafe archive member')
 archive.extractall(base)
sample=base/'platforms/s/samples/vision/resnet18';entry=sample/'runtime/cpp';build=base/'build'
report={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':SOURCE,'source_archive_sha256':DIGEST,'target':'s100'}
def run(command):
 print('COMMAND '+json.dumps(command),flush=True)
 result=subprocess.run(command,cwd=base,capture_output=True,text=True)
 print(result.stdout,flush=True);print('COMMAND_STDERR '+result.stderr,flush=True);print('COMMAND_EXIT '+str(result.returncode),flush=True)
 if result.returncode:raise SystemExit(result.returncode)
run(['cmake','-S',str(entry),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release'])
run(['cmake','--build',str(build),'--parallel','2'])
model=root/'models/resnet18_224x224_nv12.hbm'
image=sample/'test_data/zebra_cls.jpg';labels=base/'platforms/s/datasets/imagenet/imagenet_classes.names'
command=[str(build/'resnet18'),'--model_path='+str(model),'--test_img='+str(image),'--label_file='+str(labels)]
result=subprocess.run(command,cwd=base,capture_output=True,text=True)
print('COMMAND '+json.dumps(command));print(result.stdout);print(result.stderr);assert result.returncode==0
lines=[line for line in result.stdout.splitlines() if line.startswith('TOP-')];assert len(lines)==5,lines
(root/'baseline-results/resnet-cpp.json').write_text(json.dumps(lines),encoding='utf-8')
report['topk']=lines
for name,path in [('model',model),('input',image),('labels',labels),('executable',build/'resnet18')]:report[name+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
print('RESULT '+json.dumps(report),flush=True)
print('RESNET_CPP_BASELINE_PASS',flush=True)
