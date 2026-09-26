"""Capture actual exit status, timestamps and complete logs; no board access."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json, subprocess, sys
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
commands={name:[sys.executable,'-m','unittest','discover','-s',path] for name,path in {
 'ultralytics':'samples/vision/ultralytics_yolo/tests',
 'shared':'samples/_shared/tests','resnet':'samples/vision/resnet/tests',
 'ocr':'samples/vision/paddle_ocr/tests','checker':'tools/sample_contract/tests'}.items()}
commands['contracts']=[sys.executable,'tools/sample_contract/check.py','--scope','migration','--parser-mode','import','--report',str(OUT/'contracts.json')]
commands['readmes']=[sys.executable,str(OUT/'check_readmes.py')]
def run(item):
 name,argv=item
 start=datetime.now(timezone.utc).isoformat()
 with (OUT/f'{name}.log').open('w') as log:
  result=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 return dict(name=name,argv=argv,cwd=str(ROOT),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rc=result.returncode,log=f'{name}.log')
with ThreadPoolExecutor(max_workers=4) as pool:
 records=list(pool.map(run,commands.items()))
(OUT/'host-results.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
sys.exit(any(row['rc'] for row in records))
