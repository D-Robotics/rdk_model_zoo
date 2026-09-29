"""Run exact bilingual host commands against the task-local real OpenCV build."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
native=ROOT/'samples/vision/yoloe/runtime/cpp'
commands=re.findall(r'```bash\n(.*?)```',(native/'README.md').read_text(),re.S)
assert commands==re.findall(r'```bash\n(.*?)```',(native/'README_cn.md').read_text(),re.S)
env=dict(os.environ)
env['PATH']=str(ROOT.parent/'.coordination/native-build-tools/cmake/data/bin')+os.pathsep+env['PATH']
env['OpenCV_DIR']=str(ROOT.parent/'.coordination/opencv-native/install/lib/cmake/opencv4')
jobs=[('documented-native',['bash','-e','-c','\n'.join(commands)]),
 ('yoloe',[sys.executable,'-m','unittest','discover','-s','samples/vision/yoloe/tests']),
 ('contracts',[sys.executable,'tools/sample_contract/check.py','--scope','migration','--parser-mode','import','--report',str(OUT/'contracts.json')]),
 ('readmes',[sys.executable,str(OUT.parent/'2026-09-28-yoloe-evaluation/check_readmes.py')])]
records=[]
for name,argv in jobs:
 start=datetime.now(timezone.utc).isoformat()
 with (OUT/(name+'.log')).open('w') as log:result=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 records.append(dict(name=name,argv=argv,cwd=str(ROOT),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rc=result.returncode,log=name+'.log'))
(OUT/'host-results.json').write_text(json.dumps(dict(environment_overrides={'OpenCV_DIR':env['OpenCV_DIR'],'PATH_prefix':env['PATH'].split(os.pathsep)[0]},runs=records),indent=2)+'\n')
(OUT/'implementation-sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(native.rglob('*')) if p.is_file()},indent=2)+'\n')
print([(r['name'],r['rc']) for r in records]);sys.exit(any(r['rc'] for r in records))
