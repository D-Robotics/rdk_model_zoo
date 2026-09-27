"""Run the production CLI with explicitly synthetic SDK and board readers."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,math,subprocess
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent/'fixture-runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUT.mkdir(parents=True)
BINARY=ROOT.parent/'.coordination/yoloe-cli-host/tests/yoloe_cli_fixture'
MODEL=OUT/'not-a-deployable-model.fixture';MODEL.write_bytes(b'explicit host-only float result fixture\n')
IMAGE=OUT/'input.png';assert cv2.imwrite(str(IMAGE),np.full((64,64,3),(20,40,60),dtype=np.uint8))
LABELS=ROOT/'samples/vision/yoloe/test_data/classes.names'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
BASE=[str(BINARY),'--target','s100p','--variant','26n','--model-path',str(MODEL),'--model-sha256',sha(MODEL),'--test-img',str(IMAGE),'--label-file',str(LABELS),'--output',str(OUT/'positive')]
records=[]
def run(name,argv,expected):
 start=datetime.now(timezone.utc).isoformat();result=subprocess.run(argv,cwd=ROOT,capture_output=True)
 (OUT/(name+'.stdout.log')).write_bytes(result.stdout);(OUT/(name+'.stderr.log')).write_bytes(result.stderr)
 row=dict(name=name,argv=argv,cwd=str(ROOT),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rc=result.returncode,expected_rc=expected,matched=result.returncode==expected,stdout_log=name+'.stdout.log',stderr_log=name+'.stderr.log')
 records.append(row);return result
run('help',[str(BINARY),'--help'],0)
result=run('positive',BASE,0)
checks={}
if result.returncode==0:
 report=json.loads((OUT/'positive/report.json').read_text())
 item=report['instances'][0];mask=cv2.imread(str(OUT/'positive'/item['mask']),cv2.IMREAD_GRAYSCALE)
 checks=dict(backend=report['execution_backend']=='host-fixture',count=report['count']==1,class_id=item['class_id']==2,score=abs(item['score']-1/(1+math.exp(-2)))<1e-7,labels=item['label']==LABELS.read_text().splitlines()[2],image_hash=report['image_sha256']==sha(IMAGE),model_hash=report['model_sha256']==sha(MODEL),roi_shape=mask.shape==(4,4),binary_pixels=set(np.unique(mask).tolist())<={0,255},foreground=int(np.count_nonzero(mask))==9,visualization=cv2.imread(str(OUT/'positive/annotated.png')).shape==(64,64,3))
 for name,flag,value in [('wrong-target','--target','s100'),('wrong-model','--model-sha256','0'*64),('wrong-vocabulary','--label-file',str(MODEL))]:
  argv=BASE.copy();argv[argv.index(flag)+1]=value;argv[argv.index('--output')+1]=str(OUT/name);run(name,argv,2);checks[name+'-no-output']=not (OUT/name).exists()
 for name,extra in [('e26-nms',['--nms-thres','0.7']),('duplicate-option',['--score-thres','0.4','--score-thres','0.5'])]:
  argv=BASE.copy()+extra;argv[argv.index('--output')+1]=str(OUT/name);run(name,argv,2);checks[name+'-no-output']=not (OUT/name).exists()
 before=sha(OUT/'positive/report.json');run('existing-output',BASE,2);checks['no-overwrite']=before==sha(OUT/'positive/report.json')
summary=dict(host_fixture=True,board_or_sdk_evidence=False,binary_sha256=sha(BINARY),input_sha256=sha(IMAGE),model_sha256=sha(MODEL),vocabulary_sha256=sha(LABELS),runs=records,checks=checks)
(OUT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'directory':str(OUT),'runs':[(r['name'],r['rc']) for r in records],'checks':checks},indent=2))
assert len(records)==8 and all(r['matched'] for r in records) and checks and all(checks.values())
