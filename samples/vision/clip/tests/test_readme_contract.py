# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Run the exact bilingual API examples against real host adapters."""
import contextlib
import io
from pathlib import Path
import re
import shlex
import unittest
from unittest.mock import patch
import numpy as np
from test_clip import SAMPLE, ImageRuntime, TextSession


class ReadmeTests(unittest.TestCase):
    def test_bilingual_api_uses_actual_binding_runner_task_and_bpe(self):
        from samples.vision.clip.runtime.python import model_runner
        original = model_runner.RuntimeModelRunner
        for filename in ('README.md','README_cn.md'):
            snippets = re.findall(r'```python\n(.*?)```',(SAMPLE/'runtime/python'/filename).read_text(),re.S)
            self.assertEqual(len(snippets),1)
            image, text = ImageRuntime(), TextSession()
            scope = {}
            with patch.object(model_runner,'RuntimeModelRunner',lambda selection:original(selection,image_runtime=image,text_session=text)), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(snippets[0],filename,'exec'),scope)
            self.assertEqual(len(image.calls),2)
            self.assertEqual(len(text.calls),2)
            np.testing.assert_array_equal(scope['explicit_result'].scores,scope['composed_result'].scores)
            np.testing.assert_array_equal(scope['explicit_result'].order,scope['composed_result'].order)

    def test_cli_examples_parse_and_local_links_exist(self):
        from samples.vision.clip.runtime.python.main import build_parser
        from samples.vision.clip.model.download import build_parser as download_parser
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
                    options=build_parser().parse_args(args[2:]);self.assertTrue(Path(options.test_img).is_file())
                elif args[1].endswith('/model/download.py'):
                    download_parser().parse_args(args[2:])
