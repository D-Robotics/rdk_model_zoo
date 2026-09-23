# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Execute the exact bilingual importlib API and parse documented commands."""
import contextlib
import importlib
import io
from pathlib import Path
import re
import shlex
import unittest
from unittest.mock import patch
import numpy as np
from test_3dresnet import SAMPLE, FakeRuntime, local_module


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

    def test_board_recipe_keeps_evidence_and_never_auto_accepts_tied_ids(self):
        import json
        import os
        import shutil
        import sys
        import tempfile
        import types
        from test_3dresnet import ROOT
        runner_mod=local_module('model_runner')
        original=runner_mod.RuntimeModelRunner
        for filename, tied in [('README.md',False),('README_cn.md',True)]:
            text=(SAMPLE/'evaluator'/filename).read_text()
            snippets=re.findall(r"python3 - <<'PY'\n(.*?)\nPY\n",text,re.S)
            self.assertEqual(len(snippets),1)
            with tempfile.TemporaryDirectory() as directory:
                repo=Path(directory)
                for relative in ['samples/vision/3dresnet/test_data/video0.npy',
                                 'samples/vision/3dresnet/test_data/kinetics_classnames.json',
                                 'platforms/s/samples/vision/3dresnet/runtime/python/resnet3d.py']:
                    dest=repo/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(ROOT/relative,dest)
                model=repo/'samples/vision/3dresnet/model/s100/r3d_18.hbm'
                model.parent.mkdir(parents=True);model.write_bytes(b'host only')
                legacy_runtime,unified_runtime=FakeRuntime(),FakeRuntime()
                if tied:
                    legacy_runtime.raw[:]=0;unified_runtime.raw[:]=0
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
                                exec(compile(snippets[0],filename,'exec'),{})
                        else:
                            exec(compile(snippets[0],filename,'exec'),{})
                    finally:
                        fake_source=str(repo/'platforms/s')
                        if fake_source in sys.path:sys.path.remove(fake_source)
                reports=list(repo.glob('evaluator-output/**/comparison.json'))
                self.assertEqual(len(reports),1)
                report=json.loads(reports[0].read_text())
                self.assertEqual(report['passed'],not tied)
                self.assertEqual(len(list(reports[0].parent.glob('*.npy'))),4)
