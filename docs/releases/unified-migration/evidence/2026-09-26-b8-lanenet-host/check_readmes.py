"""Check real README links, paired commands, parsers and Python API on a host.

Parser checks stop immediately after parsing: no downloads, SDK or OE calls.
The API executes with the actual runner and explicitly injected fake SDK.
"""
import argparse
import importlib
import inspect
import json
from pathlib import Path
import re
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
SAMPLE = ROOT / 'samples/vision/lanenet'

class Parsed(BaseException):
    pass

links = 0
commands = []
api_runs = []
for en in sorted(SAMPLE.rglob('README.md')):
    zh = en.with_name('README_cn.md')
    assert zh.is_file(), zh
    paired = []
    for page in (en, zh):
        content = page.read_text()
        prose = re.sub(r'```.*?```', '', content, flags=re.S)
        for target in re.findall(r'\]\(([^)]+)\)', prose):
            target = target.split('#')[0]
            if not target or '://' in target:
                continue
            assert (page.parent / target).exists(), (page, target)
            links += 1
        bash = re.findall(r'```bash\n(.*?)```', content, re.S)
        shell_commands = []
        for block in bash:
            for line in block.replace('\\\n', ' ').splitlines():
                words = shlex.split(line, comments=True)
                if words:
                    shell_commands.append(words)
        paired.append(shell_commands)
    assert paired[0] == paired[1], en
    for words in paired[0]:
        module = None
        args = None
        if words[0] in ('python', 'python3'):
            if words[1] == '-m':
                module, args = words[2], words[3:]
            else:
                script = ROOT / words[1] if words[1].startswith('samples/') else en.parent / words[1]
                module = '.'.join(script.relative_to(ROOT).with_suffix('').parts)
                args = words[2:]
        elif words[0] == 'bash':
            script = ROOT / words[1]
            if script.name == 'download.sh':
                module = 'samples.vision.lanenet.model.download'
            elif 'runtime/python/' in words[1]:
                module = 'samples.vision.lanenet.runtime.python.main'
            elif 'runtime/cpp/' in words[1]:
                module = 'samples.vision.lanenet.runtime.cpp.launcher'
            args = words[2:]
        if module == 'unittest':
            continue  # Executed separately as the complete sample suite.
        if module:
            entry = importlib.import_module(module)
            original = argparse.ArgumentParser.parse_args
            def stop_after_parse(parser, argv=None, namespace=None):
                original(parser, args if argv is None else argv, namespace)
                raise Parsed()
            with patch.object(argparse.ArgumentParser, 'parse_args', stop_after_parse):
                try:
                    if inspect.signature(entry.main).parameters:
                        entry.main(args)
                    else:
                        entry.main()
                except Parsed:
                    pass
                else:
                    raise AssertionError(('parser not reached', module))
            commands.append({'page':str(en.relative_to(ROOT)), 'argv':words})

from samples.vision.lanenet.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.lanenet.tests.test_lanenet import metadata, raw_outputs
for name in ('README.md', 'README_cn.md'):
    page = SAMPLE / 'runtime/python' / name
    snippet = re.findall(r'```python\n(.*?)```', page.read_text(), re.S)[0]
    m = metadata(True)
    raw = raw_outputs(True)
    runtime = SimpleNamespace(**{k:v for k,v in m.items() if k != 'model_name'},
        run=lambda inputs: {'lane': raw}, set_scheduling_params=lambda **kw: None)
    with patch('samples.vision.lanenet.runtime.python.model_runner.RuntimeModelRunner',
               side_effect=lambda s: RuntimeModelRunner(s, runtime=runtime)):
        context = {}
        exec(compile(snippet, str(page), 'exec'), context)
        np.testing.assert_array_equal(context['result'].embedding, raw['instance_seg_logits'][0])
        np.testing.assert_array_equal(context['result'].binary, raw['binary_seg_pred'].reshape(256,512))
        api_runs.append(str(page.relative_to(ROOT)))
print(json.dumps({'local_links':links,'parser_commands':commands,'api_fixture_runs':api_runs,
                  'real_sdk':'not-run','board':'not-run'}, indent=2))
