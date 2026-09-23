"""Structural bilingual documentation checks."""
import re, shlex, unittest, contextlib, io, json
from pathlib import Path
from samples.vision.yoloworld.runtime.python.main import build_parser
from samples.vision.yoloworld.tests.test_yoloworld import FakeRuntime
from samples.vision.yoloworld.model.download import build_parser as download_parser
from unittest.mock import patch
SAMPLE=Path(__file__).resolve().parents[1]
class ReadmeTests(unittest.TestCase):
    def test_runtime_api_snippet_executes_with_injected_fixture(self):
        from samples.vision.yoloworld.runtime.python import model_runner
        original = model_runner.RuntimeModelRunner
        for filename in ("README.md", "README_cn.md"):
            text = (SAMPLE / "runtime/python" / filename).read_text(encoding="utf-8")
            snippets = re.findall(r"```python\n(.*?)```", text, re.S)
            self.assertEqual(len(snippets), 1)
            runtime = FakeRuntime()
            def factory(selection, _runtime=runtime):
                return original(selection, runtime=_runtime)
            with patch.object(model_runner, "RuntimeModelRunner", factory), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(snippets[0], filename, "exec"), {})
            self.assertGreaterEqual(len(runtime.calls), 2)

    def test_ten_readmes_have_required_sections_and_local_links(self):
        files=list(SAMPLE.rglob('README*.md'));self.assertEqual(len(files),10)
        required=('support-matrix','prerequisites','quickstart','expected-results','directory','entry-points')
        for path in files:
            text=path.read_text(encoding='utf-8').replace('\\\n',' ')
            self.assertTrue(all(re.search(r'^#+ .*'+re.escape(x),text,re.M|re.I) for x in required if path.parent==SAMPLE),path)
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if '://' not in target: self.assertTrue((path.parent/target.split('#')[0]).exists(),(path,target))
            for line in text.splitlines():
                if line.startswith('python3 samples/') or line.startswith('.venv/bin/python samples/'):
                    args=shlex.split(line); cmd=args[1]
                    if cmd.endswith('/runtime/python/main.py'): build_parser().parse_args(args[2:])
                    elif cmd.endswith('/model/download.py'): download_parser().parse_args(args[2:])
if __name__=='__main__': unittest.main()
