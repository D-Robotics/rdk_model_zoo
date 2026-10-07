# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Execute the exact bilingual importlib API and parse documented commands."""
import contextlib
import importlib
import importlib.util
import io
from pathlib import Path
import re
import shlex
import unittest
from samples._shared.tests.legacy_platforms import legacy_path  # noqa: E402
from unittest.mock import patch
import numpy as np
from test_3dresnet import SAMPLE, FakeRuntime, local_module


def comparison_fixture():
    """Load the test-only parity recipe removed from the customer-facing README."""
    path=Path(__file__).resolve().parent/'fixtures/migration_comparison.py'
    spec=importlib.util.spec_from_file_location('migration_comparison_3dresnet',path)
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReadmeTests(unittest.TestCase):
    def test_bilingual_api_uses_qualified_imports_and_actual_pipeline(self):
        runner_mod=local_module('model_runner')
        original=runner_mod.RuntimeModelRunner
        for filename in ('README.md','README_cn.md'):
            text=(SAMPLE/'runtime/python'/filename).read_text()
            snippets=re.findall(r'```python\n(.*?)```',text,re.S)
            self.assertEqual(len(snippets),1)
            runtime=FakeRuntime()
            scope={}
            with patch.object(runner_mod,'RuntimeModelRunner',lambda selection:original(selection,runtime=runtime)),contextlib.redirect_stdout(io.StringIO()):
                exec(compile(snippets[0],filename,'exec'),scope)
            self.assertEqual(len(runtime.calls),2)
            a,b=scope['explicit_result'],scope['composed_result']
            np.testing.assert_array_equal(a.class_ids,b.class_ids)
            np.testing.assert_array_equal(a.scores,b.scores)
            self.assertEqual(a.labels,b.labels)
            self.assertEqual(len(a.labels),5)

    def test_native_commands_parse_and_all_local_links_resolve(self):
        parser=local_module('main').build_parser()
        download_parser=importlib.import_module('samples.vision.3dresnet.model.download').build_parser()
        files=list(SAMPLE.rglob('README*.md'))
        self.assertEqual(len(files),10)
        for path in files:
            text=path.read_text().replace('\\\n',' ')
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if '://' not in target:
                    self.assertTrue((path.parent/target.split('#')[0]).exists(),(path,target))
            for line in text.splitlines():
                if not line.startswith('python3 samples/'):continue
                args=shlex.split(line)
                if args[1].endswith('/runtime/python/main.py'):
                    options=parser.parse_args(args[2:])
                    self.assertTrue(Path(options.test_clip).is_file())
                    self.assertTrue(Path(options.label_file).is_file())
                elif args[1].endswith('/model/download.py'):
                    download_parser.parse_args(args[2:])

    def test_documented_shell_download_commands_delegate_valid_arguments_without_network(self):
        import os
        import subprocess
        import tempfile
        parser=importlib.import_module('samples.vision.3dresnet.model.download').build_parser()
        with tempfile.TemporaryDirectory() as directory:
            shim=Path(directory)/'python3'
            shim.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            shim.chmod(0o755)
            environment={**os.environ,'PATH':directory+os.pathsep+os.defpath}
            count=0
            for path in SAMPLE.rglob('README*.md'):
                for line in path.read_text().replace('\\\n',' ').splitlines():
                    if not line.startswith('bash samples/vision/3dresnet/model/download.sh'):continue
                    argv=shlex.split(line)
                    result=subprocess.run(argv,cwd=SAMPLE.parents[2],env=environment,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,(path,result.stderr))
                    forwarded=result.stdout.splitlines()
                    self.assertTrue(forwarded[0].endswith('/download.py'))
                    with contextlib.redirect_stderr(io.StringIO()):
                        options=parser.parse_args(forwarded[1:])
                    self.assertEqual(options.target,'s100')
                    count+=1
            self.assertGreaterEqual(count,8)

    def test_migration_comparison_fixture_keeps_evidence_and_never_auto_accepts_tied_ids(self):
        """Test-only source-parity check, not documented usage: execute the
        legacy/unified comparison recipe preserved verbatim in
        tests/fixtures/migration_comparison.py (the heredoc the customer-facing
        evaluator rewrite removed) under a fake SDK and temporary repository."""
        import json
        import os
        import shutil
        import sys
        import tempfile
        import types
        from test_3dresnet import ROOT
        runner_mod=local_module('model_runner')
        original=runner_mod.RuntimeModelRunner
        fixture=comparison_fixture()
        for variant, tied in [('README_MD',False),('README_CN_MD',True)]:
            snippet=getattr(fixture,variant)
            self.assertTrue(snippet.strip(),variant)
            filename=f'fixtures/migration_comparison.py:{variant}'
            with tempfile.TemporaryDirectory() as directory:
                repo=Path(directory)
                fixtures=['samples/vision/3dresnet/test_data/video0.npy',
                          'samples/vision/3dresnet/test_data/kinetics_classnames.json']
                for relative in fixtures:
                    dest=repo/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(ROOT/relative,dest)
                # The historical source file now lives only in Git; seed the
                # fixture repo's platforms/s copy from the pinned commit.
                legacy='platforms/s/samples/vision/3dresnet/runtime/python/resnet3d.py'
                dest=repo/legacy;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(legacy_path('s/samples/vision/3dresnet/runtime/python/resnet3d.py'),dest)
                model=repo/'samples/vision/3dresnet/model/s100/r3d_18.hbm'
                model.parent.mkdir(parents=True);model.write_bytes(b'host only')
                legacy_runtime,unified_runtime=FakeRuntime(),FakeRuntime()
                if tied:
                    # The fixture must force a real Top-K ID mismatch on every
                    # platform. Zeroing both sides (the previous fixture) relied
                    # on numpy's unspecified unstable-argsort tie order: the
                    # source helper's default argsort returned a different order
                    # than the unified stable sort on macOS/arm64 but the same
                    # first five on HP's Linux build (2026-09-23), where the IDs
                    # then matched, the recipe correctly passed and the expected
                    # assertion was never raised. Instead, the unified side gets
                    # five classes nudged 1e-7 above the tie: raw outputs stay
                    # inside the recipe's atol=1e-5 while the unified Top-5 is
                    # deterministically [100..104] against the source side's
                    # order over all-equal zeros.
                    legacy_runtime.raw[:]=0
                    unified_runtime.raw[:]=0;unified_runtime.raw[0,100:105]=np.float32(1e-7)
                    legacy_ids=[int(i) for i in np.argsort(-np.full(400,0.0025,dtype=np.float32))[:5]]
                    bump=np.zeros(400,dtype=np.float32);bump[100:105]=np.float32(1e-7)
                    probs=np.exp(bump.astype(np.float64)-float(bump.max()));probs/=probs.sum()
                    unified_ids=[int(i) for i in np.argsort(-probs,kind='stable')[:5]]
                    self.assertNotEqual(legacy_ids,unified_ids,'tied-id fixture no longer forces a Top-K ID mismatch')
                sdk=types.ModuleType('hbm_runtime')
                sdk.HB_HBMRuntime=lambda path:legacy_runtime
                sdk.QuantParams=object
                # The real source helper is already imported by the source tests.
                from test_3dresnet import load_legacy_source
                load_legacy_source()
                out=repo/'evaluator-output/fixture-run';out.mkdir(parents=True)
                with patch.dict(os.environ,{'OUT_DIR':str(out)}),patch.object(Path,'cwd',return_value=repo),patch.dict(sys.modules,{'hbm_runtime':sdk}),patch.object(runner_mod,'RuntimeModelRunner',lambda selection:original(selection,runtime=unified_runtime)),patch('samples._shared.platforms.require_execution_target',return_value='s100'),contextlib.redirect_stdout(io.StringIO()):
                    try:
                        if tied:
                            with self.assertRaisesRegex(AssertionError,'No automatic tie exemption'):
                                exec(compile(snippet,filename,'exec'),{})
                        else:
                            exec(compile(snippet,filename,'exec'),{})
                    finally:
                        fake_source=str(repo/'platforms/s')
                        if fake_source in sys.path:sys.path.remove(fake_source)
                reports=list(repo.glob('evaluator-output/**/comparison.json'))
                self.assertEqual(len(reports),1)
                report=json.loads(reports[0].read_text())
                self.assertEqual(report['passed'],not tied)
                self.assertEqual(len(list(reports[0].parent.glob('*.npy'))),4)
