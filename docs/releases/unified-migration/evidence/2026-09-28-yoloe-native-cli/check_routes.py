"""Exercise only host selection commands; never invoke a board or SDK."""
from datetime import datetime, timezone
from pathlib import Path
import json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
LAUNCHER=ROOT/'samples/vision/yoloe/runtime/cpp/launcher.py'
from samples.vision.yoloe.runtime.python.model_binding import list_models
records=[]
def run(name,args,rc=0):
    start=datetime.now(timezone.utc).isoformat()
    p=subprocess.run([sys.executable,str(LAUNCHER),*args],cwd=ROOT,capture_output=True)
    (OUT/f'{name}.stdout.log').write_bytes(p.stdout)
    (OUT/f'{name}.stderr.log').write_bytes(p.stderr)
    records.append(dict(name=name,argv=p.args,cwd=str(ROOT),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rc=p.returncode,expected_rc=rc))
    assert p.returncode==rc,(name,p.stderr)
    return p
p=run('list-models',['--list-models'])
assert len(json.loads(p.stdout))==14
for target,variant,asset in list_models('auto'):
    p=run(f'dry-{target}-{variant}',['--target',target,'--asset-id',asset.reference,'--dry-run'])
    data=json.loads(p.stdout)
    assert (data['target'],data['variant'])==(target,variant)
    assert not any(data[k] for k in ('executed','downloaded','runtime_metadata_verified'))
    assert not data['processes']
    if target!='x5':assert data['status']=='requires local floating-output conversion'
run('unsupported-s600',['--target','s600','--dry-run'],2)
run('auto-dry-run',['--dry-run'],2)
run('published-s-execution',['--target','s100p'],2)
# Import isolation: this subprocess has no image/SDK imports through the launcher.
p=subprocess.run([sys.executable,'-c',"import sys; from samples.vision.yoloe.runtime.cpp import launcher; assert 'cv2' not in sys.modules; assert 'hbm_runtime' not in sys.modules"],cwd=ROOT,capture_output=True)
assert p.returncode==0,p.stderr
(OUT/'route-results.json').write_text(json.dumps(dict(runs=records,import_isolation={'argv':p.args,'rc':p.returncode},board_or_sdk_executed=False),indent=2)+'\n')
print(f'{len(records)} host CLI cases and import isolation passed')
