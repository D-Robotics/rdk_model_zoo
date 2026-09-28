"""Reproduce Paraformer binding tests and executable bilingual README commands."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
checks = []
for directory, log, expected in [('samples/speech/paraformer/tests', 'green.log', 23), ('samples/_shared/tests', 'shared.log', 156)]:
    arguments = (
        [f'samples.speech.paraformer.tests.test_{name}' for name in ('cif', 'pipeline', 'binding', 'download')]
        if directory == 'samples/speech/paraformer/tests'
        else ['discover', '-s', directory]
    )
    result = subprocess.run([sys.executable, '-m', 'unittest', *arguments, '-v'], cwd=ROOT, capture_output=True, text=True)
    (HERE / log).write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    count = int(re.search(r'Ran (\d+) tests?', result.stderr).group(1))
    assert count == expected, (directory, count)
    checks.append({'suite':directory, 'tests':count, 'rc':result.returncode})
examples = []
for filename in ('README.md', 'README_cn.md'):
    path = ROOT / 'samples/speech/paraformer/runtime/python' / filename
    content = path.read_text()
    for index, block in enumerate(re.findall(r'```bash\n(.*?)```', content, re.S)[:4]):
        env = dict(os.environ)
        env['PATH'] = str(Path(sys.executable).parent) + ':' + env['PATH']
        result = subprocess.run(['bash', '-c', block], cwd=ROOT, env=env, capture_output=True, text=True)
        (HERE / f'{filename}-block-{index}.log').write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stderr
        if index == 0: assert result.stdout.strip() == '(1, 100, 512) [2] [3.0, 8.0]'
        if index == 2: assert result.stdout.strip() == '中中文 3 True'
        if index == 3:
            lines = result.stdout.splitlines()
            assert len(lines) == 4
            assert lines[0] == 'encoder s:paraformer:s100/paraformer_large_encoder_400x560_s100.hbm'
            assert lines[1] == 'predictor s:paraformer:s100/paraformer_large_predictor_400x512_s100.hbm'
            assert lines[2] == 'decoder s:paraformer:s100/paraformer_large_decoder_400x512_s100.hbm'
            assert 'only for s100' in lines[3]
        examples.append({'readme':filename, 'block':index, 'rc':result.returncode})
    for link in re.findall(r'\]\(([^)]+)\)', content):
        assert (path.parent / link).resolve().exists(), link
auxiliary = []
for filename in ('am.mvn', 'paraformer_config.yaml'):
    local = ROOT / 'samples/speech/paraformer/model' / filename
    source = subprocess.check_output(['git', 'show', '380e1a2bf42041af54be6f34935e50197cfadff9:samples/speech/paraformer/model/' + filename], cwd=ROOT)
    assert local.read_bytes() == source
    auxiliary.append({'file':filename, 'matches_source':True})
for filename in ('README.md', 'README_cn.md'):
    path = ROOT / 'samples/speech/paraformer/model' / filename
    content = path.read_text()
    block = re.findall(r'```bash\n(.*?)```', content, re.S)[0]
    env = dict(os.environ)
    env['PYTHON'] = sys.executable
    result = subprocess.run(['bash', '-c', block], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0 and len(result.stdout.splitlines()) == 6
    (HERE / f'model-{filename}-preview.log').write_text(result.stdout + result.stderr)
    for link in re.findall(r'\]\(([^)]+)\)', content):
        assert (path.parent / link).resolve().exists(), link
help_result = subprocess.run([sys.executable, str(ROOT / 'samples/speech/paraformer/model/download.py'), '--help'], cwd=ROOT, capture_output=True, text=True)
assert help_result.returncode == 0
(HERE / 'download-help.log').write_text(help_result.stdout + help_result.stderr)
summary = {'scope':'host binding and shared-runner SDK doubles, not real SDK or board execution', 'checks':checks, 'examples':examples, 'board_api_sketch':'not-run','source_auxiliaries':auxiliary,'model_readme_previews':2,'download_help_rc':help_result.returncode, 'python':sys.version}
(HERE / 'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'checks':checks,'documented_commands':len(examples)}))
