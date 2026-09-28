"""Reproduce source failure, execute README examples, and retain host test output."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE_PATH = 'platforms/s/samples/speech/paraformer/conversion/cif_numpy.py'
source = ROOT / SOURCE_PATH
source_bytes = source.read_bytes()
pinned = subprocess.check_output(
    ['git', 'show', '380e1a2bf42041af54be6f34935e50197cfadff9:samples/speech/paraformer/conversion/cif_numpy.py'], cwd=ROOT
)
assert pinned == source_bytes, 'Archived source differs from fixed S commit'
spec = importlib.util.spec_from_file_location('source_cif', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    module.cif_numpy(np.zeros((1,401), np.float32), np.zeros((1,401,512), np.float32), real_T=400)
except IndexError as exc:
    failure = {'type': type(exc).__name__, 'message': str(exc)}
else:
    raise AssertionError('Expected original zero-fire failure did not reproduce')
checks = []
for name in ('README.md', 'README_cn.md'):
    path = ROOT / 'samples/speech/paraformer/runtime/python' / name
    content = path.read_text()
    for i, block in enumerate(re.findall(r'```bash\n(.*?)```', content, re.S)[:2]):
        # Use the active Python interpreter for the documented python command.
        env = dict(__import__('os').environ)
        env['PATH'] = str(Path(sys.executable).parent) + ':' + env['PATH']
        result = subprocess.run(['bash', '-c', block], cwd=ROOT, env=env, capture_output=True, text=True)
        (HERE / f'{name}-example-{i}.log').write_text(result.stdout + result.stderr)
        assert result.returncode == 0, (path, i, result.stderr)
        if i == 0:
            assert result.stdout.strip() == '(1, 100, 512) [2] [3.0, 8.0]'
        checks.append({'readme': str(path.relative_to(ROOT)), 'block': i, 'rc': result.returncode})
    for target in re.findall(r'\]\(([^)]+)\)', content):
        if target.startswith('#'): continue
        assert (path.parent / target).resolve().exists(), target
result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'samples/speech/paraformer/tests', '-p', 'test_cif.py', '-v'], cwd=ROOT, capture_output=True, text=True)
(HERE / 'green.log').write_text(result.stdout + result.stderr)
assert result.returncode == 0, result.stderr
summary = {
    'scope': 'host-only CIF numerical bridge; no model, SDK, board or conversion execution',
    'source_commit': '380e1a2bf42041af54be6f34935e50197cfadff9',
    'source_path': SOURCE_PATH,
    'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
    'source_matches_git': True,
    'source_empty_fire_failure': failure,
    'test_rc': result.returncode,
    'test_count': 7,
    'normal_source_comparisons': 24,
    'readme_examples': checks,
    'python': sys.version,
    'numpy': np.__version__,
}
(HERE / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(summary, indent=2, ensure_ascii=False))
