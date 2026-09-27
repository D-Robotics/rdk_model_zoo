"""Execute documented host checks; preserve commands and complete diagnostics."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
page=ROOT/'samples/vision/yoloe/runtime/cpp/README.md'
cn=ROOT/'samples/vision/yoloe/runtime/cpp/README_cn.md'
commands=re.findall(r'```bash\n(.*?)```',page.read_text(),re.S)
assert commands==re.findall(r'```bash\n(.*?)```',cn.read_text(),re.S)
jobs=[('documented-native-tests',['bash','-e','-c','\n'.join(commands)]),
 ('yoloe',[sys.executable,'-m','unittest','discover','-s','samples/vision/yoloe/tests']),
 ('contracts',[sys.executable,'tools/sample_contract/check.py','--scope','migration','--parser-mode','import','--report',str(OUT/'contracts.json')]),
 ('readmes',[sys.executable,str(OUT.parent/'2026-09-28-yoloe-evaluation/check_readmes.py')])]
records=[]
for name,argv in jobs:
 start=datetime.now(timezone.utc).isoformat()
 with (OUT/f'{name}.log').open('w') as log:r=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 records.append(dict(name=name,argv=argv,cwd=str(ROOT),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rc=r.returncode,log=name+'.log'))
(OUT/'host-results.json').write_text(json.dumps(records,indent=2)+'\n')
files=list((ROOT/'samples/vision/yoloe/runtime/cpp').rglob('*'))
files+=list((ROOT/'samples/vision/ultralytics_yolo/runtime/cpp/common').glob('*.h'))
(OUT/'implementation-sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files) if p.is_file()},indent=2)+'\n')
print([(r['name'],r['rc']) for r in records])
sys.exit(any(r['rc'] for r in records))
