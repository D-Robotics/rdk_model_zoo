"""Execute current bilingual native build/API examples with real JSON headers."""
from pathlib import Path
import re, subprocess, os, json, tempfile
root=Path.cwd(); here=Path(__file__).resolve().parent
runtime=root/'samples/speech/paraformer/runtime/cpp'
en=(runtime/'README.md').read_text(); cn=(runtime/'README_cn.md').read_text()
blocks=re.findall(r'```bash\n(.*?)```',en,re.S)
assert blocks==re.findall(r'```bash\n(.*?)```',cn,re.S) and len(blocks)==3
cpp=re.findall(r'```cpp\n(.*?)```',en,re.S)
assert cpp==re.findall(r'```cpp\n(.*?)```',cn,re.S) and len(cpp)==2
env=dict(os.environ)
env['PATH']=str(root.parent/'.coordination/native-build-tools/cmake/data/bin')+os.pathsep+env['PATH']
env['CMAKE_PREFIX_PATH']=str(root.parent/'.coordination/asr-json')
records=[]
for index,block in enumerate(blocks):
    result=subprocess.run(['bash','-euc',block],cwd=root,env=env,capture_output=True,text=True)
    (here/f'doc-{index}.log').write_text(result.stdout+result.stderr)
    records.append({'command':block,'cwd':str(root),'returncode':result.returncode,'CMAKE_PREFIX_PATH':env['CMAKE_PREFIX_PATH']})
    assert result.returncode==0,result.stdout+result.stderr
    if index in (0,2):
        count=4 if index==0 else 5
        assert re.search(rf'100% tests passed(?:, 0 tests failed)? out of {count}',result.stdout)
    else: assert result.stdout.strip()=='2 3 8'
with tempfile.TemporaryDirectory(prefix='paraformer-io-doc-') as temp:
    for index,source in enumerate(cpp):
        path=Path(temp)/f'example-{index}.cc';path.write_text(source)
        argv=['c++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(runtime/'inc'),'-I'+str(root/'samples/_shared/cpp'),'-c',str(path),'-o',str(path.with_suffix('.o'))]
        result=subprocess.run(argv,capture_output=True,text=True)
        (here/f'api-{index}.log').write_text(result.stdout+result.stderr)
        records.append({'argv':argv,'returncode':result.returncode})
        assert result.returncode==0,result.stdout+result.stderr
(here/'doc-summary.json').write_text(json.dumps(records,indent=2)+'\n')
print('Three bilingual shell examples and both complete API functions passed')
