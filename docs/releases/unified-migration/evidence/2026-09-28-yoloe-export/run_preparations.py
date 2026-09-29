"""Prepare all target routes from actual exports, using one smoke image (not calibration acceptance)."""
from datetime import datetime,timezone
from pathlib import Path
import json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from samples.vision.yoloe.runtime.python.model_binding import list_models
EXPORTS=ROOT.parent/'.coordination/yoloe-export-final-v3-20260928'
BUILD=ROOT.parent/'.coordination/yoloe-prepare-real-20260928'
BUILD.mkdir(exist_ok=False)
records=[]
for target,variant,asset in list_models():
    case=target+'-'+variant
    argv=[sys.executable,'samples/vision/yoloe/conversion/prepare.py','--onnx',str(EXPORTS/variant/f'yoloe_{variant}_seg_pf.onnx'),'--names',str(EXPORTS/variant/f'yoloe_{variant}_seg_pf.names'),'--target',target,'--variant',variant,'--cal-images','samples/vision/yoloe/test_data','--sample-count','1','--output-dir',str(BUILD/case)]
    record={'case':case,'argv':argv,'cwd':str(ROOT),'started_utc':datetime.now(timezone.utc).isoformat(),'log':case+'-prepare.log'}
    with (OUT/record['log']).open('w') as log:
        process=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    record.update(returncode=process.returncode,finished_utc=datetime.now(timezone.utc).isoformat())
    if process.returncode==0:
        result=json.loads((BUILD/case/'conversion.json').read_text())
        assert result['status']=='config_only' and result['compiler'] is None
        assert not any('attention node absent' in w for w in result['warnings'])
        (OUT/(case+'-prepare.json')).write_text(json.dumps(result,indent=2)+'\n')
        shutil.copyfile(BUILD/case/'config.yaml',OUT/(case+'-config.yaml'))
        shutil.copyfile(BUILD/case/'calibration.json',OUT/(case+'-calibration.json'))
    records.append(record)
    (OUT/'real-preparation-results.json').write_text(json.dumps(records,indent=2)+'\n')
    print(case,process.returncode,flush=True)
sys.exit(any(r['returncode'] for r in records))
