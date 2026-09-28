import subprocess,json,datetime,os,re
from pathlib import Path
r=Path('/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk-b7-board-integration');out=r/'docs/releases/unified-migration/evidence/2026-09-29-independent-closeout';py=str(r.parent/'rdk_model_zoo/.venv/bin/python');records=[]
env={**os.environ,'PATH':str(Path(py).parent)+os.pathsep+os.environ['PATH'],'PYTHONDONTWRITEBYTECODE':'1'}
paths=[r/'samples/_shared/tests',*sorted((r/'samples').glob('*/*/tests')),r/'tools/sample_contract/tests',r/'tools/board_validation/tests']
for p in paths:
 if 'gemma4-e2b' in p.parts or not list(p.glob('test*.py')):continue
 name='suite-'+str(p.relative_to(r)).replace('/','-');argv=[py,'-m','unittest','discover','-s',str(p.relative_to(r)),'-v'];start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 result=subprocess.run(argv,cwd=r,env=env,capture_output=True,text=True)
 (out/(name+'.stdout.log')).write_text(result.stdout);(out/(name+'.stderr.log')).write_text(result.stderr)
 summary=re.findall(r'Ran \d+ tests?[^\n]*|OK(?: \([^\n]*\))?|FAILED[^\n]*',result.stderr)
 records.append(dict(id=name,argv=argv,cwd=str(r),started_at=start,ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=result.returncode,summary=summary))
 (out/'host-integration.json').write_text(json.dumps(records,indent=2)+'\n');print(name,result.returncode,summary[-2:],flush=True)
argv=[py,'tools/sample_contract/check.py','--scope','migration','--format','json'];p=subprocess.run(argv,cwd=r,env=env,capture_output=True,text=True);(out/'migration-contract.json').write_text(p.stdout);(out/'migration-contract.stderr.log').write_text(p.stderr);print('contract',p.returncode,flush=True)
