"""Compile the native probe and capture the actual real-ONNX comparison run."""
from pathlib import Path
from datetime import datetime, timezone
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
commands=[('compiler',['c++','--version']),('build-probe',['c++','-std=c++17','-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-omit-frame-pointer','-I','samples/vision/yoloe/runtime/cpp/common','samples/vision/yoloe/runtime/cpp/tests/decode_probe.cc','-o',str(ROOT.parent/'.coordination/yoloe-decode-probe')]),('compare',[sys.executable,str(OUT/'compare_onnx.py')])]
rows=[]
for name,argv in commands:
 start=datetime.now(timezone.utc).isoformat()
 with (OUT/f'{name}.log').open('w') as stream:r=subprocess.run(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
 rows.append(dict(name=name,argv=argv,cwd=str(ROOT),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rc=r.returncode,log=name+'.log'))
 if r.returncode:break
(OUT/'comparison-execution.json').write_text(json.dumps(rows,indent=2)+'\n')
print([(r['name'],r['rc']) for r in rows]);sys.exit(any(r['rc'] for r in rows))
