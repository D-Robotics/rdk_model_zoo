"""Host-only pipeline tests, documented examples and source decoder comparison."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from samples.speech.paraformer.runtime.python.decoding import decode_logits

vocab_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'published-tokens.json'
raw = vocab_path.read_bytes()
vocabulary = json.loads(raw)
assert hashlib.sha256(raw).hexdigest() == '2b20c2b12572d682afff84ce1c8d560f67b8b32a4c1f21567411d141ed352127'
assert len(vocabulary) == len(set(vocabulary)) == 8404
source_path = 'platforms/s/samples/speech/paraformer/runtime/python/paraformer.py'
source = (ROOT / source_path).read_bytes()
pinned = subprocess.check_output(['git','show','380e1a2bf42041af54be6f34935e50197cfadff9:samples/speech/paraformer/runtime/python/paraformer.py'], cwd=ROOT)
assert source == pinned
# Compile only the original numerical post_process method: no SDK/Torch import,
# no reimplementation of its text algorithm in this verification script.
source_class = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Paraformer')
method = next(n for n in source_class.body if isinstance(n, ast.FunctionDef) and n.name == 'post_process')
namespace = {'np': np}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / source_path), 'exec'), namespace)
comparisons = []
for seed in range(5):
    logits = np.random.default_rng(seed).normal(size=(1,100,8404)).astype(np.float32)
    for count in (0,1,37,100):
        actual, ids = decode_logits(logits, count, vocabulary)
        expected = namespace['post_process'](SimpleNamespace(vocab=vocabulary), logits, np.array([count], np.int32))
        assert actual == expected
        comparisons.append({'seed':seed, 'count':count, 'text_sha256':hashlib.sha256(actual.encode()).hexdigest(), 'equal':True})
examples=[]
for filename in ('README.md','README_cn.md'):
    path=ROOT/'samples/speech/paraformer/runtime/python'/filename
    content=path.read_text()
    for index, block in enumerate(re.findall(r'```bash\n(.*?)```',content,re.S)[:3]):
        env=dict(os.environ);env['PATH']=str(Path(sys.executable).parent)+':'+env['PATH']
        p=subprocess.run(['bash','-c',block],cwd=ROOT,env=env,capture_output=True,text=True)
        (HERE/f'{filename}-block-{index}.log').write_text(p.stdout+p.stderr)
        assert p.returncode==0,p.stderr
        if index==0: assert p.stdout.strip()=='(1, 100, 512) [2] [3.0, 8.0]'
        if index==2: assert p.stdout.strip()=='中中文 3 True'
        examples.append({'readme':filename,'block':index,'rc':p.returncode})
    for link in re.findall(r'\]\(([^)]+)\)',content):
        if link.startswith('#'): continue
        assert (path.parent/link).resolve().exists(),link
p=subprocess.run([sys.executable,'-m','unittest','samples.speech.paraformer.tests.test_cif','samples.speech.paraformer.tests.test_pipeline','-v'],cwd=ROOT,capture_output=True,text=True)
(HERE/'green.log').write_text(p.stdout+p.stderr)
assert p.returncode==0,p.stderr
summary={'scope':'CPU-only orchestration with synthetic model callables; no SDK or board inference',
 'tests':13,'test_rc':p.returncode,'source_sha256':hashlib.sha256(source).hexdigest(),
 'source_matches_git':True,'text_source_comparisons':comparisons,'examples':examples,
 'vocabulary':{'url':'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/paraformer/nash-e/tokens.json','size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'tokens':len(vocabulary),'unique':len(set(vocabulary))},
 'python':sys.version,'numpy':np.__version__}
(HERE/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(f'{len(comparisons)} source text comparisons; 13 tests; {len(examples)} documented commands passed')
