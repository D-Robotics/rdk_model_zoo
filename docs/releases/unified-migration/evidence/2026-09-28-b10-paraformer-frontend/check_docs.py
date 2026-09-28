"""Execute host README examples; skip explicit installs and package downloads."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
seen={}
records=[]
for folder in ('runtime/python','model','test_data'):
    for filename in ('README.md','README_cn.md'):
        path=ROOT/'samples/speech/paraformer'/folder/filename
        content=path.read_text()
        for target in re.findall(r'\]\(([^)]+)\)',content):
            assert (path.parent/target).resolve().exists(),target
        for i,block in enumerate(re.findall(r'```bash\n(.*?)```',content,re.S)):
            if ' -m venv ' in block or (folder=='model' and '--dry-run' not in block):
                records.append({'path':str(path.relative_to(ROOT)),'block':i,'status':'not-replayed-explicit-setup-or-download'})
                continue
            key=hashlib.sha256(block.encode()).hexdigest()
            if key not in seen:
                env=dict(os.environ);env['PATH']=str(Path(sys.executable).parent)+':'+env['PATH'];env['PYTHON']=sys.executable
                p=subprocess.run(['bash','-c',block],cwd=ROOT,env=env,capture_output=True,text=True)
                logfile=f'{folder.replace("/","-")}-{filename}-{i}.log'
                (HERE/logfile).write_text(p.stdout+p.stderr)
                assert p.returncode==0,p.stderr
                if 'frontend.pre_process' in block:
                    assert '(1, 400, 560) 71 71 False' in p.stdout
                seen[key]={'rc':p.returncode,'log':logfile}
            records.append({'path':str(path.relative_to(ROOT)),'block':i,**seen[key]})
manifest=ROOT/'samples/speech/paraformer/test_data/manifest.json'
source=subprocess.check_output(['git','show','380e1a2bf42041af54be6f34935e50197cfadff9:samples/speech/paraformer/test_data/manifest.json'],cwd=ROOT)
assert manifest.read_bytes()==source
summary={'interpreter':sys.executable,'commands':records,'unique_executed_commands':len(seen),'manifest_matches_source':True,'shared_identical_blocks':'executed once and reused across languages; not counted as separate executions'}
(HERE/'docs-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(f'{len(seen)} distinct host commands passed; matching bilingual blocks verified')
