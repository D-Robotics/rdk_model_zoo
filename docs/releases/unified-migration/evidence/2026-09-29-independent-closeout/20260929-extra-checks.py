import subprocess,json,datetime,os,re
from pathlib import Path
r=Path('/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk-b7-board-integration');out=r/'docs/releases/unified-migration/evidence/2026-09-29-independent-closeout';base=r.parent/'.coordination';py=str(r.parent/'rdk_model_zoo/.venv/bin/python');cmake=str(base/'native-build-tools/cmake/data/bin/cmake');ctest=str(Path(cmake).with_name('ctest'));records=[]
env={**os.environ,'PATH':str(Path(py).parent)+os.pathsep+os.environ['PATH'],'PYTHONPATH':os.pathsep.join(str(base/x) for x in ['yoloe-conversion-deps','yoloe-evaluator-deps','yoloe-export-deps']),'PYTHONDONTWRITEBYTECODE':'1'}
checks=[('yoloe-all-host',[py,'-m','unittest','discover','-s','samples/vision/yoloe/tests','-v'],env),('yoloe-evaluator',[py,'-m','unittest','discover','-s','samples/vision/yoloe/evaluator/tests','-v'],env),('yoloe-export-synthetic',[py,'-m','unittest','discover','-s','samples/vision/yoloe/conversion/tests','-v'],env),('paraformer-optional-host',[str(base/'paraformer-frontend-venv/bin/python'),'-m','unittest','discover','-s','samples/speech/paraformer/tests','-v'],{**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})]
for name in ['yoloe-stage-library','ultralytics-final-independent-native']:
 checks += [(name+'-build',[cmake,'--build',str(base/name),'-j','4'],env),(name+'-ctest',[ctest,'--test-dir',str(base/name),'--output-on-failure'],env)]
for name,argv,e in checks:
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(argv,cwd=r,env=e,text=True,capture_output=True)
 (out/(name+'.stdout.log')).write_text(p.stdout);(out/(name+'.stderr.log')).write_text(p.stderr)
 records.append(dict(id=name,argv=argv,cwd=str(r),pythonpath=e.get('PYTHONPATH'),started_at=start,ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=p.returncode))
 (out/'extra-checks.json').write_text(json.dumps(records,indent=2)+'\n');print(name,p.returncode,(p.stdout+p.stderr)[-200:],flush=True)
