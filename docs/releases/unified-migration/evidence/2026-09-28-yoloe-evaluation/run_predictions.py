from pathlib import Path
import json,hashlib,subprocess,sys
from datetime import datetime,timezone
root=Path.cwd();base=root.parent/'.coordination/yoloe-evaluation-final'
base.mkdir(exist_ok=False)
labels=(root/'samples/vision/yoloe/test_data/classes.names').read_text().splitlines()
from samples.vision.yoloe.model.vocabulary import LABELS_SHA256
from samples.vision.yoloe.runtime.python.model_binding import list_models
import cv2
image=root/'samples/vision/yoloe/test_data/office_desk.jpg';h,w=cv2.imread(str(image)).shape[:2]
annotation={'images':[{'id':1,'file_name':image.name,'height':h,'width':w}],'categories':[{'id':i+1,'name':name} for i,name in enumerate(labels)],'annotations':[]}
mapping={'vocabulary_sha256':LABELS_SHA256,'mapping':[{'pf_id':i,'pf_name':name,'category_id':i+1,'category_name':name} for i,name in enumerate(labels)]}
(base/'annotations.json').write_text(json.dumps(annotation));(base/'mapping.json').write_text(json.dumps(mapping))
rows=[]
for target,variant,_ in list_models():
 case=target+'-'+variant
 model=root.parent/'.coordination/yoloe-export-final-v3-20260928'/variant/f'yoloe_{variant}_seg_pf.onnx'
 digest=hashlib.sha256(model.read_bytes()).hexdigest()
 argv=[sys.executable,'samples/vision/yoloe/evaluator/evaluate.py','--backend','onnx','--target',target,'--variant',variant,'--model-path',str(model),'--model-sha256',digest,'--image-dir',str(image.parent),'--annotation',str(base/'annotations.json'),'--category-map',str(base/'mapping.json'),'--output-dir',str(base/case),'--predictions-only']
 row={'case':case,'argv':argv,'cwd':str(root),'started_utc':datetime.now(timezone.utc).isoformat(),'log':str(base/(case+'.log'))}
 with Path(row['log']).open('w') as log:
  process=subprocess.run(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 row.update(returncode=process.returncode,finished_utc=datetime.now(timezone.utc).isoformat())
 rows.append(row);(base/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(case,process.returncode,flush=True)
sys.exit(any(r['returncode'] for r in rows))
