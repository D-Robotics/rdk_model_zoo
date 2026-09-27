"""Capture catalog checks with actual status; source manifests are already updated."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess,json,os
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
env=os.environ.copy();env['PATH']='/opt/homebrew/opt/node@22/bin:'+env['PATH']
records=[]
for name,argv in [('publisher-tests',['npm','--prefix','tools/catalog-publisher','test']),('publisher-build',['npm','--prefix','tools/catalog-publisher','run','build'])]:
 start=datetime.now(timezone.utc).isoformat()
 with (OUT/f'{name}.log').open('w') as log:
  process=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 records.append(dict(name=name,argv=argv,cwd=str(ROOT),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rc=process.returncode,log=f'{name}.log'))
(OUT/'catalog-results.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
raise SystemExit(any(row['rc'] for row in records))
