"""Capture public YOLOE-26 release sidecars; never fetch/run HBM or connect to a board."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import urlopen
import hashlib,json
OUT=Path(__file__).resolve().parent
BASE='https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/yoloe26_seg'
records=[]
def fetch(march,name,expected=None):
    url=f'{BASE}/{march}/{name}'
    row={'url':url,'started_utc':datetime.now(timezone.utc).isoformat(),'march':march,'name':name}
    try:
        with urlopen(url,timeout=20) as response:
            data=response.read(5*1024*1024+1)
            if len(data)>5*1024*1024:raise ValueError('Sidecar exceeds size limit')
        digest=hashlib.sha256(data).hexdigest()
        if expected is not None:
            if expected['sha256']!=digest or expected['bytes']!=len(data):raise ValueError('Sidecar digest/length mismatch')
        path=OUT/'release'/march/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        row.update(status='captured',sha256=digest,bytes=len(data),file=str(path.relative_to(OUT)),verified_against_manifest=expected is not None)
    except Exception as error:
        row.update(status='failed',error=f'{type(error).__name__}: {error}')
    row['finished_utc']=datetime.now(timezone.utc).isoformat()
    return row
with ThreadPoolExecutor(max_workers=2) as pool:
    manifests=list(pool.map(lambda march:fetch(march,'manifest.json'),('nash-e','nash-m')))
records.extend(manifests)
tasks=[]
for row in manifests:
    if row['status']!='captured':continue
    data=json.loads((OUT/row['file']).read_text())
    if data.get('march')!=row['march']:raise ValueError('Manifest target mismatch')
    for size in 'nsmlx':
        for suffix in ('.json','.names'):
            name=f'yoloe_26{size}_seg_pf{suffix}'
            tasks.append((row['march'],name,data['files'][name]))
with ThreadPoolExecutor(max_workers=4) as pool:
    records.extend(pool.map(lambda args:fetch(*args),tasks))
(OUT/'release-capture.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps({'captured':sum(r['status']=='captured' for r in records),'failed':sum(r['status']=='failed' for r in records),'hbm_downloaded':False}))
raise SystemExit(any(r['status']=='failed' for r in records))
