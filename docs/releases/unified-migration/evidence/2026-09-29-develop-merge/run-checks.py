import subprocess,json,os,datetime
from pathlib import Path
r=Path('/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk_model_zoo');b=r.parent/'.coordination';d=b/'20260929-develop-merge-checks';d.mkdir(exist_ok=True);old=r/'docs/releases/unified-migration/evidence/2026-09-29-independent-closeout';py=str(r/'.venv/bin/python');records=[]
env={**os.environ,'PATH':str(r/'.venv/bin')+':/opt/homebrew/opt/node@22/bin:'+os.environ['PATH'],'PYTHONDONTWRITEBYTECODE':'1'}
def check(name,args,e=env):
 p=subprocess.run(args,cwd=r,env=e,capture_output=True,text=True);(d/(name+'.stdout.log')).write_text(p.stdout);(d/(name+'.stderr.log')).write_text(p.stderr);records.append({'id':name,'argv':args,'cwd':str(r),'exit_code':p.returncode,'pythonpath':e.get('PYTHONPATH')});(d/'checks.json').write_text(json.dumps(records,indent=2)+'\n');print(name,p.returncode,flush=True)
check('catalog',['npm','--prefix','tools/catalog-publisher','run','check'])
for x in json.loads((old/'host-integration.json').read_text()):
 args=x['argv'];e=env.copy()
 if 'paraformer' in args[-2]:args[0]=str(b/'paraformer-frontend-venv/bin/python')
 if 'yoloe' in args[-2]:e['PYTHONPATH']=':'.join(str(b/n) for n in ['yoloe-conversion-deps','yoloe-evaluator-deps','yoloe-export-deps'])
 check(x['id'],args,e)
for name,path in [('gemma','samples/llm/gemma4-e2b/tests'),('skills','skills/tests'),('yoloe-evaluator','samples/vision/yoloe/evaluator/tests'),('yoloe-export','samples/vision/yoloe/conversion/tests')]:
 e=env.copy()
 if name.startswith('yoloe'):e['PYTHONPATH']=':'.join(str(b/n) for n in ['yoloe-conversion-deps','yoloe-evaluator-deps','yoloe-export-deps'])
 check(name,[py,'-m','unittest','discover','-s',path,'-v'],e)
check('contract',[py,'tools/sample_contract/check.py','--scope','migration','--format','json'])
check('skills-pack',[py,'skills/tools/validate_pack.py','--pack-root','skills'])
check('skills-references',[py,'skills/tools/sync_references.py'])
print('FAILED',[x['id'] for x in records if x['exit_code']],flush=True)
