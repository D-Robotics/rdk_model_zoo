"""Execute the two identical bilingual host examples from repository root."""
from pathlib import Path
import re
import subprocess
import os
import json

root = Path.cwd()
evidence = Path(__file__).resolve().parent
runtime = root / 'samples/speech/paraformer/runtime/cpp'
blocks = re.findall(r'```bash\n(.*?)```', (runtime / 'README.md').read_text(), re.S)
assert blocks == re.findall(r'```bash\n(.*?)```', (runtime / 'README_cn.md').read_text(), re.S)
assert len(blocks) == 2
env = dict(os.environ)
env['PATH'] = str(root.parent / '.coordination/native-build-tools/cmake/data/bin') + os.pathsep + env['PATH']
records = []
for index, command in enumerate(blocks):
    completed = subprocess.run(['bash', '-euc', command], cwd=root, env=env, capture_output=True, text=True)
    (evidence / f'doc-{index}.log').write_text(completed.stdout + completed.stderr)
    records.append({'command': command, 'cwd': str(root), 'rc': completed.returncode})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    if index == 1:
        assert completed.stdout.strip() == '2 3 8', completed.stdout
(evidence / 'doc-summary.json').write_text(json.dumps(records, indent=2) + '\n')
print('Two bilingual native README examples passed; numerical output: 2 3 8')
