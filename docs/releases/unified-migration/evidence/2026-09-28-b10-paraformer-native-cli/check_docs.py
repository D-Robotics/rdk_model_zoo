"""Execute current host README paths; explicitly exclude download/board commands."""
from pathlib import Path
import hashlib, json, os, re, subprocess, tempfile
root=Path.cwd();here=Path(__file__).resolve().parent;runtime=root/'samples/speech/paraformer/runtime/cpp'
en=(runtime/'README.md').read_text();cn=(runtime/'README_cn.md').read_text()
blocks=re.findall(r'```bash\n(.*?)```',en,re.S);assert blocks==re.findall(r'```bash\n(.*?)```',cn,re.S) and len(blocks)==6
cpp=re.findall(r'```cpp\n(.*?)```',en,re.S);assert cpp==re.findall(r'```cpp\n(.*?)```',cn,re.S) and len(cpp)==2
env=dict(os.environ)
env['PATH']=str(root.parent/'.coordination/paraformer-frontend-venv/bin')+os.pathsep+str(root.parent/'.coordination/native-build-tools/cmake/data/bin')+os.pathsep+env['PATH']
env['CMAKE_PREFIX_PATH']=str(root.parent/'.coordination/asr-json')
records=[]
for index,block in enumerate(blocks):
    if 'download_model.sh --target s100' in block:
        records.append({'index':index,'command':block,'status':'not-run; model download and board environment required'});continue
    with tempfile.TemporaryDirectory(prefix='paraformer-doc-cwd-') as temporary:
        cwd=root
        if '--preprocess-only' in block:
            cwd=Path(temporary);(cwd/'samples').symlink_to(root/'samples',target_is_directory=True)
        result=subprocess.run(['bash','-euc',block],cwd=cwd,env=env,capture_output=True,text=True)
        (here/f'doc-{index}.log').write_text(result.stdout+result.stderr)
        record={'index':index,'command':block,'cwd':str(cwd),'returncode':result.returncode,'python':str(root.parent/'.coordination/paraformer-frontend-venv/bin/python'),'CMAKE_PREFIX_PATH':env['CMAKE_PREFIX_PATH']}
        records.append(record)
        assert result.returncode==0,result.stdout+result.stderr
        if index in (3,5):
            count=4 if index==3 else 6
            assert re.search(rf'100% tests passed(?:, 0 tests failed)? out of {count}',result.stdout)
        elif index==4:assert result.stdout.strip()=='2 3 8'
        elif index==1:
            prepared=cwd/'outputs/paraformer_features'
            entries=json.loads((prepared/'prepared-manifest.json').read_text())
            old=root/'docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-cli/run-5_sjkdzc/features'
            for entry in entries:
                data=(prepared/entry['feature_file']).read_bytes()
                assert data==(old/entry['feature_file']).read_bytes()
                assert hashlib.sha256(data).hexdigest()==entry['feature_sha256']
            record['prepared_entries']=entries
            record['real_frontend_result']=json.loads((prepared/'result.json').read_text())
with tempfile.TemporaryDirectory(prefix='paraformer-cli-api-') as temporary:
    for index,code in enumerate(cpp):
        src=Path(temporary)/f'example-{index}.cc';src.write_text(code)
        argv=['c++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(runtime/'inc'),'-I'+str(root/'samples/_shared/cpp'),'-c',str(src),'-o',str(src.with_suffix('.o'))]
        result=subprocess.run(argv,capture_output=True,text=True);(here/f'api-{index}.log').write_text(result.stdout+result.stderr)
        records.append({'argv':argv,'returncode':result.returncode});assert result.returncode==0,result.stderr
(here/'doc-summary.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
print('Five host README shell blocks, real FunASR preparation and two C++ API examples passed; board/download block not run')
