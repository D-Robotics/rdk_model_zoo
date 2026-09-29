"""Check current local links and run the pure API example, never hub downloads."""
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
result = {}
for directory in ('', 'conversion'):
    for language in ('README.md', 'README_cn.md'):
        path = ROOT / 'samples/speech/paraformer' / directory / language
        text = path.read_text()
        links = 0
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith(('https:', 'http:', '#')):
                continue
            assert (path.parent / target.split('#')[0]).exists(), (path, target)
            links += 1
        examples = 0
        for code in re.findall(r'```python\n(.*?)```', text, re.S):
            if 'gather_indices_int32' in code:
                exec(compile(code, str(path), 'exec'), {})
                examples += 1
        result[str(path.relative_to(ROOT))] = {'local_links': links, 'graph_api_examples': examples}
command = [sys.executable, 'samples/speech/paraformer/conversion/export.py', '--help']
run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
for flag in ('--model-dir', '--output-dir', '--feature', '--threads'):
    assert flag in run.stdout
result['export_help'] = {'rc': run.returncode, 'argv': command}
result['not_executed_by_this_checker'] = ['dependency installation', 'hub download', 'full export (separate evidence)']
Path(__file__).with_name('docs-check.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
