"""Run production pure helpers and narrow SDK-descriptor doubles with sanitizers."""
from pathlib import Path
import subprocess,json,datetime
root=Path(__file__).resolve().parents[5];out=Path(__file__).resolve().parent
cpp=root/'samples/vision/ultralytics_yolo/runtime/cpp'
records=[]
for name in ['test_decode','test_head_probe','test_nv12_geometry','test_benchmark','test_classification','test_classification_binding_X5','test_classification_binding_UCP','test_task_outputs','test_task_output_binding_X5','test_task_output_binding_UCP']:
 argv=['c++','-std=c++11','-Wall','-Wextra','-fsanitize=address,undefined']
 if 'binding' in name:
  argv+=['-I',str(cpp/('test/fake_task_outputs' if 'task_output_binding' in name else 'test/fake_classification')),'-DYOLO_DNN_STACK_'+name.rsplit('_',1)[1]+'=1']
  source='test_task_output_binding' if 'task_output_binding' in name else 'test_classification_binding'
 else:source=name
 argv+=['-I',str(cpp),str(cpp/f'test/{source}.cc'),str(cpp/'common/nv12_geometry.cc'),str(cpp/'common/benchmark.cc'),'-o','/tmp/rdk-yolo-'+name]
 for stage,command in [('compile',argv),('run',['/tmp/rdk-yolo-'+name])]:
  start=datetime.datetime.now(datetime.timezone.utc).isoformat()
  result=subprocess.run(command,cwd=root,capture_output=True,text=True)
  log=f'{name}-{stage}.log';(out/log).write_text(result.stdout+result.stderr)
  records.append({'name':name,'stage':stage,'argv':command,'cwd':str(root),'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rc':result.returncode,'log':log})
  if result.returncode:break
(out/'native-results.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
raise SystemExit(any(r['rc'] for r in records))
