"""Run bilingual host commands and compile the actual public SDK API example."""
from pathlib import Path
import re
import subprocess
import os
import json
import tempfile
root = Path.cwd()
evidence = Path(__file__).resolve().parent
runtime = root / 'samples/speech/paraformer/runtime/cpp'
en = (runtime / 'README.md').read_text()
cn = (runtime / 'README_cn.md').read_text()
blocks = re.findall(r'```bash\n(.*?)```', en, re.S)
assert blocks == re.findall(r'```bash\n(.*?)```', cn, re.S) and len(blocks) == 2
cpp = re.findall(r'```cpp\n(.*?)```', en, re.S)
assert cpp == re.findall(r'```cpp\n(.*?)```', cn, re.S) and len(cpp) == 1
env = dict(os.environ)
env['PATH'] = str(root.parent / '.coordination/native-build-tools/cmake/data/bin') + os.pathsep + env['PATH']
records = []
def run(command, label, expected=0):
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    (evidence / (label + '.log')).write_text(result.stdout + result.stderr)
    records.append({'argv':command, 'cwd':str(root), 'returncode':result.returncode, 'expected':expected})
    assert (result.returncode == 0) == (expected == 0), result.stdout + result.stderr
    return result
for index, block in enumerate(blocks):
    result = run(['bash', '-euc', block], f'doc-{index}')
    if index == 0:
        assert re.search(r'100% tests passed(?:, 0 tests failed)? out of 3', result.stdout)
    else:
        assert result.stdout.strip() == '2 3 8'
with tempfile.TemporaryDirectory(prefix='paraformer-sdk-doc-') as temporary:
    tmp = Path(temporary)
    (tmp / 'example.cc').write_text(cpp[0])
    run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I'+str(runtime/'inc'), '-c', str(tmp/'example.cc'), '-o', str(tmp/'example.o')], 'api-compile')
    result = run(['cmake', '-S', str(runtime), '-B', str(tmp/'sdk'), '-DPARAFORMER_BUILD_SDK=ON'], 'vendor-sdk-configure', expected=1)
    assert 'PARAFORMER_DNN_INCLUDE' in result.stdout + result.stderr
(evidence/'doc-summary.json').write_text(json.dumps(records, indent=2)+'\n')
print('Two bilingual commands passed; three CTests and SDK API compilation passed; absent real SDK rejected')
