# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Execute customer API snippets and verify source-pinned conversion commands."""
import ast
import contextlib
import importlib.util
import io
from pathlib import Path
import re
import shlex
import unittest
from unittest.mock import patch
import numpy as np
from test_dinov2 import FakeRuntime, ROOT, SAMPLE
from utils.py_utils.tests.legacy_platforms import legacy_path, legacy_tree  # noqa: E402


def comparison_fixture():
    """Load the test-only parity recipe removed from the customer-facing README."""
    path = Path(__file__).resolve().parent / 'fixtures/migration_comparison.py'
    spec = importlib.util.spec_from_file_location('migration_comparison_dinov2', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReadmeTests(unittest.TestCase):
    def test_runtime_api_examples_execute_real_binding_and_runner(self):
        from samples.vision.dinov2.runtime.python import embedding as runner_mod
        original = runner_mod.RuntimeModelRunner
        for filename in ('README.md', 'README_cn.md'):
            snippets = re.findall(r'```python\n(.*?)```', (SAMPLE/'runtime/python'/filename).read_text(), re.S)
            self.assertEqual(len(snippets), 1, filename)
            runtime = FakeRuntime('I16')
            scope = {}
            with patch.object(runner_mod, 'RuntimeModelRunner', lambda selection: original(selection, runtime=runtime)), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(snippets[0], filename, 'exec'), scope)
            self.assertEqual(scope['composed_result'].shape, (1, 384))
            self.assertEqual(scope['composed_result'].dtype, np.float32)
            self.assertEqual(len(runtime.calls), 2)
            np.testing.assert_array_equal(scope['explicit_result'], scope['composed_result'])

    def test_native_commands_parse_and_all_local_links_resolve(self):
        from samples.vision.dinov2.runtime.python.main import build_parser
        from samples.vision.dinov2.model.download import build_parser as download_parser
        files = list(SAMPLE.rglob('README*.md'))
        self.assertEqual(len(files), 10)
        for path in files:
            text = path.read_text().replace('\\\n', ' ')
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if '://' not in target:
                    self.assertTrue((path.parent/target.split('#')[0]).exists(), (path, target))
            for line in text.splitlines():
                if not line.startswith('python3 samples/'):
                    continue
                args = shlex.split(line)
                if args[1].endswith('/runtime/python/main.py'):
                    options = build_parser().parse_args(args[2:])
                    self.assertTrue(Path(options.test_img).is_file())
                elif args[1].endswith('/model/download.py'):
                    download_parser().parse_args(args[2:])

    def test_checkpoint_digest_and_conversion_scripts_match_fixed_source(self):
        files = ['mapper.py', 'onnx_export/export_dinov2.py']
        source = legacy_tree('s/samples/vision/dinov2/conversion')
        digests = set()
        for rel in files:
            code = (SAMPLE/'conversion'/rel).read_bytes()
            self.assertEqual(code, (source/rel).read_bytes())
            tree = ast.parse(code)
            for statement in tree.body:
                if isinstance(statement, ast.Assign) and any(isinstance(name, ast.Name) and name.id == 'WEIGHTS_SHA256' for name in statement.targets):
                    digests.add(ast.literal_eval(statement.value))
        self.assertEqual(len(digests), 1)
        digest = digests.pop()
        self.assertRegex(digest, r'^[0-9a-f]{64}$')
        for filename in ('README.md', 'README_cn.md'):
            text = (SAMPLE/'conversion'/filename).read_text()
            arguments = re.findall(r'--weights-sha256\s+([0-9a-f]+)', text)
            self.assertGreaterEqual(len(arguments), 2)
            self.assertEqual(set(arguments), {digest})

    def test_migration_comparison_fixture_parity_with_host_runtime_fixtures(self):
        """Test-only source-parity check, not documented usage: execute the
        legacy/unified comparison recipe preserved verbatim in
        tests/fixtures/migration_comparison.py (the heredoc the customer-facing
        evaluator rewrite removed) under fake SDK modules and a temporary
        repository."""
        import json
        import shutil
        import sys
        import tempfile
        from test_dinov2 import load_legacy_source
        from samples.vision.dinov2.runtime.python import cli as model_binding
        from samples.vision.dinov2.runtime.python import embedding as model_runner
        original_resolve = model_binding.resolve_selection
        original_runner = model_runner.RuntimeModelRunner
        legacy_module = load_legacy_source()
        fixture = comparison_fixture()
        for variant, mismatch in [('README_MD',False),('README_CN_MD',True)]:
            snippet = getattr(fixture, variant)
            self.assertTrue(snippet.strip(), variant)
            filename = f'fixtures/migration_comparison.py:{variant}'
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                image=root/'samples/vision/dinov2/test_data/dog.jpg'
                image.parent.mkdir(parents=True)
                shutil.copyfile(SAMPLE/'test_data/dog.jpg',image)
                model=root/'fixture.hbm';model.write_bytes(b'host fixture only')
                unified=FakeRuntime('I16')
                old=FakeRuntime('I16');old.output_quants={'dinov2':old.output_quants}
                if mismatch:unified.raw['cls_feat'][0,0]+=1
                def selection(*args,**kwargs):
                    kwargs.setdefault('model_path',model)
                    kwargs.setdefault('asset_id','s:dinov2:nash-e/dinov2_vits14_224_int16_nashe.hbm')
                    return original_resolve(*args,**kwargs)
                with patch.object(Path,'cwd',return_value=root), patch.dict(sys.modules,{'dinov2':legacy_module}), patch.object(legacy_module.hbm_runtime,'HB_HBMRuntime',return_value=old), patch.object(model_binding,'resolve_selection',selection), patch.object(model_runner,'RuntimeModelRunner',lambda chosen:original_runner(chosen,runtime=unified)), patch('utils.py_utils.platforms.require_execution_target',return_value='s100'),contextlib.redirect_stdout(io.StringIO()):
                    if mismatch:
                        with self.assertRaisesRegex(AssertionError,'migration parity failed'):
                            exec(compile(snippet,filename,'exec'),{})
                    else:
                        exec(compile(snippet,filename,'exec'),{})
                reports=list((root/'evaluator-output').glob('*/comparison.json'))
                self.assertEqual(len(reports),1)
                report=json.loads(reports[0].read_text())
                self.assertEqual(report['passed'],not mismatch)
                self.assertEqual(len(list(reports[0].parent.glob('*.npy'))),9)
