"""Read-only HEAD checks of manifest CLS URLs and historical 640 aliases."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from samples._shared.assets import list_assets

assets=[a for a in list_assets('s','ultralytics_yolo') if '_cls_' in a.filename and a.filename.startswith(('nash-e/','nash-m/')) and a.filename.split('/')[-1].startswith(('yolov8','yolo11'))]
jobs=[(a.reference,kind,url) for a in assets for kind,url in [('manifest',a.url),('historical_640_alias',a.url.replace('_224x224_','_640x640_'))]]

def probe(job):
    asset,kind,url=job
    record={'asset_id':asset,'kind':kind,'requested_url':url,'method':'HEAD','started_utc':datetime.now(timezone.utc).isoformat()}
    try:
        with urlopen(Request(url,method='HEAD',headers={'User-Agent':'ModelZoo-host-audit/1.0'}),timeout=15) as response:
            record.update(status=response.status,final_url=response.url,headers={key:response.headers.get(key) for key in ('Content-Length','Content-Type','ETag','Last-Modified','Accept-Ranges')})
    except HTTPError as exc:
        record.update(status=exc.code,final_url=exc.url,error=str(exc))
    except (URLError,TimeoutError,OSError) as exc:
        record.update(status=None,error=str(exc))
    record['finished_utc']=datetime.now(timezone.utc).isoformat()
    return record

with ThreadPoolExecutor(max_workers=4) as pool:
    records=list(pool.map(probe,jobs))
print(json.dumps({'checked_utc':datetime.now(timezone.utc).isoformat(),'manifest_assets':len(assets),'records':records,'body_downloaded':False,'limitations':'HTTP availability/headers do not prove tensor dimensions, byte identity or board execution.'},indent=2))
