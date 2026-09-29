"""Check changed guides' local links and execute only new source dry-run examples."""
from pathlib import Path
import json
import os
import re
import shlex
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[5]
SAMPLE = ROOT / 'samples/vision/ultralytics_yolo'
links = []
for directory in ('', 'model', 'runtime/python'):
    for suffix in ('', '_cn'):
        page = SAMPLE / directory / f'README{suffix}.md'
        for href in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)', page.read_text()):
            if href.startswith(('http:', 'https:', 'mailto:')):
                continue
            raw, _, anchor = href.partition('#')
            target = (page.parent / unquote(raw)).resolve() if raw else page
            assert target.exists(), (page, href)
            if anchor == 'standalone-assets':
                assert f'id="{anchor}"' in target.read_text()
            links.append({'page':str(page.relative_to(ROOT)), 'href':href})
blocks = []
for suffix in ('', '_cn'):
    content = (SAMPLE / 'model' / f'README{suffix}.md').read_text().split('<a id="standalone-assets"></a>')[1]
    blocks.append(re.findall(r'```bash\n(.*?)```', content, re.S))
assert blocks[0] == blocks[1]
runs = []
env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH',''))
for block in blocks[0]:
    for line in block.replace('\\\n',' ').splitlines():
        argv = shlex.split(line)
        if not argv:
            continue
        assert '--dry-run' in argv
        if argv[0] == 'python':
            argv[0] = sys.executable
        result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
        runs.append({'argv':argv,'rc':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        assert result.returncode == 0, runs[-1]
print(json.dumps({'local_links':links,'source_dry_run_examples':runs,'board':'not-run'},indent=2))
