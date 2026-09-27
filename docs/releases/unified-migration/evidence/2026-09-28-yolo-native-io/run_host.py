"""Reproduce baseline defects, verify production replacements, retain all outputs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,os,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
CPP=ROOT/'samples/vision/ultralytics_yolo/runtime/cpp'
cmake=ROOT.parent/'.coordination/native-build-tools/cmake/data/bin'
env=dict(os.environ);env['PATH']=str(cmake)+os.pathsep+env['PATH']
records=[]
def run(name,argv,expected=0):
 start=datetime.now(timezone.utc).isoformat()
 with (OUT/(name+'.log')).open('w') as stream:
  p=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
 return dict(name=name,argv=argv,cwd=str(ROOT),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rc=p.returncode,expected_rc=expected,matched=p.returncode==expected,log=name+'.log')
for version,source in [('baseline',OUT/'baseline-dnn_io.cc'),('fixed',CPP/'common/dnn_io.cc')]:
 binary=ROOT.parent/'.coordination'/('yolo-io-regression-'+version)
 compile_cmd=['c++','-std=c++11','-fsanitize=address,undefined','-fno-omit-frame-pointer','-I',str(CPP/'test/fake_dnn_io/x5'),'-I',str(CPP),'-I',str(CPP/'common'),str(OUT/'regression.cc'),str(source),str(CPP/'common/nv12_geometry.cc'),'-o',str(binary)]
 row=run(version+'-compile',compile_cmd);records.append(row)
 if row['rc']==0:
  for case,args in [('task',[]),('capacity',['capacity'])]:records.append(run(version+'-verified-'+case,[str(binary)]+args,-6 if version=='baseline' else 0))
en=(CPP/'README.md').read_text();cn=(CPP/'README_cn.md').read_text()
host=lambda s:next(b for b in re.findall(r'```bash\n(.*?)```',s,re.S) if 'ctest --test-dir /tmp/ultralytics-cpp-host' in b)
assert host(en)==host(cn)
records.append(run('documented-native',['bash','-e','-c',host(en)]))
paths={'yoloe':'samples/vision/yoloe/tests','ultralytics':'samples/vision/ultralytics_yolo/tests','shared':'samples/_shared/tests','resnet':'samples/vision/resnet/tests','ocr':'samples/vision/paddle_ocr/tests','checker':'tools/sample_contract/tests','export':'samples/vision/yoloe/conversion/tests','evaluator':'samples/vision/yoloe/evaluator/tests'}
jobs=[(name,[sys.executable,'-m','unittest','discover','-s',path]) for name,path in paths.items()]
jobs.extend([('contracts',[sys.executable,'tools/sample_contract/check.py','--scope','migration','--parser-mode','import','--report',str(OUT/'contracts.json')]),('readmes',[sys.executable,str(OUT.parent/'2026-09-28-yoloe-evaluation/check_readmes.py')])])
with ThreadPoolExecutor(max_workers=4) as pool:records.extend(pool.map(lambda item:run(*item),jobs))
(OUT/'host-results.json').write_text(json.dumps(records,indent=2)+'\n')
files=[ROOT/'samples/vision/ultralytics_yolo/tests/test_cpp_contract.py',CPP/'common/dnn_io.h',CPP/'common/dnn_io.cc',CPP/'common/nv12_geometry.cc',CPP/'common/nv12_geometry.h',CPP/'test/test_dnn_io.cc',CPP/'test/CMakeLists.txt',CPP/'README.md',CPP/'README_cn.md']+list((CPP/'test/fake_dnn_io').rglob('*.h'))
(OUT/'implementation-sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},indent=2)+'\n')
print([(r['name'],r['rc'],r['matched']) for r in records]);sys.exit(any(not r['matched'] for r in records))
