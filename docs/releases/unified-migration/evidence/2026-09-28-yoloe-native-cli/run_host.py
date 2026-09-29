"""Capture native/readme/Python verification; deliberately never access boards."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
NATIVE=ROOT/'samples/vision/yoloe/runtime/cpp';YOLO=ROOT/'samples/vision/ultralytics_yolo/runtime/cpp'
env=dict(os.environ);env['PYTHON']=sys.executable;env['PATH']=str(ROOT.parent/'.coordination/native-build-tools/cmake/data/bin')+os.pathsep+env['PATH'];env['OpenCV_DIR']=str(ROOT.parent/'.coordination/opencv-native/install/lib/cmake/opencv4')
readmes=[(NATIVE/name).read_text() for name in ('README.md','README_cn.md')]
blocks=[re.findall(r'```bash\n(.*?)```',s,re.S) for s in readmes];assert blocks[0]==blocks[1]
api=[re.findall(r'```cpp\n(.*?)```',s,re.S) for s in readmes];assert api[0]==api[1] and len(api[0])==2
(OUT/'readme-api.cc').write_text('\n'.join(api[0]))
def run(name,argv,expected=0):
 start=datetime.now(timezone.utc).isoformat()
 with (OUT/(name+'.log')).open('w') as stream:p=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
 matched=p.returncode==expected
 if name=='sdk-dependency-gate':matched=matched and 'YOLOE SDK adapter requires board dnn headers/library' in (OUT/(name+'.log')).read_text()
 return dict(name=name,argv=argv,cwd=str(ROOT),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rc=p.returncode,expected_rc=expected,matched=matched,log=name+'.log')
commands=[('documented-native',['bash','-e','-c','\n'.join(blocks[0])]),('api',['c++','-std=c++17','-Wall','-Wextra','-Werror','-I',str(NATIVE/'inc'),'-I',str(NATIVE/'common'),'-I',str(YOLO),'-I',str(ROOT/'samples/_shared/cpp'),'-I',str(ROOT.parent/'.coordination/opencv-native/install/include/opencv4'),'-c',str(OUT/'readme-api.cc'),'-o',str(ROOT.parent/'.coordination/yoloe-sdk-readme.o')]),('shared-native',['bash','-e','-c','cmake -S samples/vision/ultralytics_yolo/runtime/cpp/test -B /tmp/ultralytics-cpp-host\ncmake --build /tmp/ultralytics-cpp-host --parallel 4\nctest --test-dir /tmp/ultralytics-cpp-host --output-on-failure']),('sdk-dependency-gate',['cmake','-S',str(NATIVE),'-B',str(ROOT.parent/'.coordination/yoloe-cli-missing-dependencies'),'-DYOLOE_BUILD_CLI=ON'],1)]
paths={'yolov5':'samples/vision/yolov5/tests','yoloe':'samples/vision/yoloe/tests','ultralytics':'samples/vision/ultralytics_yolo/tests','shared':'samples/_shared/tests','resnet':'samples/vision/resnet/tests','ocr':'samples/vision/paddle_ocr/tests','checker':'tools/sample_contract/tests','export':'samples/vision/yoloe/conversion/tests','evaluator':'samples/vision/yoloe/evaluator/tests'}
commands.extend((name,[sys.executable,'-m','unittest','discover','-s',path]) for name,path in paths.items())
commands.extend([('contracts',[sys.executable,'tools/sample_contract/check.py','--scope','migration','--parser-mode','import','--report',str(OUT/'contracts.json')]),('readmes',[sys.executable,str(OUT.parent/'2026-09-28-yoloe-evaluation/check_readmes.py')])])
with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(lambda args:run(*args),commands))
(OUT/'host-results.json').write_text(json.dumps(dict(environment={'OpenCV_DIR':env['OpenCV_DIR'],'PATH_prefix':env['PATH'].split(os.pathsep)[0]},runs=records),indent=2)+'\n')
files=[p for p in NATIVE.rglob('*') if p.is_file()]+list((ROOT/'samples/_shared/cpp').glob('*'))+[ROOT/'samples/vision/yolov5/runtime/cpp/src/yolov5_dump.cpp',ROOT/'samples/vision/yoloe/tests/test_cpp_preflight.py']+[YOLO/'common/task_output_binding.h',YOLO/'common/dnn_io.h',YOLO/'common/dnn_io.cc',YOLO/'common/dnn_resources.h']+list((YOLO/'test/fake_dnn_io').rglob('*.h'))
(OUT/'implementation-sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},indent=2)+'\n')
print([(r['name'],r['rc'],r['matched']) for r in records]);sys.exit(any(not r['matched'] for r in records))
