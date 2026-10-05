# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Canonical stage names and the thin-entry structure for the CLIP sample.

The readable-runtime design requires the model file to expose
``preprocess``/``infer``/``postprocess``/``predict`` as the primary stage
API with ``pre_process``/``forward``/``post_process`` kept as compatibility
delegates, and the entry to parse arguments through a local ``cli`` module
while visibly constructing the task and calling ``predict``.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_clip import ImageRuntime, TextSession, fixture


class CanonicalStageTests(unittest.TestCase):
    def test_canonical_stage_names_exist_and_delegate(self):
        task, _, image, text = fixture()
        pic = np.zeros((31, 72, 3), np.uint8)
        prompts = ['a diagram', 'a dog']

        prepared = task.preprocess(pic, prompts)
        legacy = task.pre_process(pic, prompts)
        np.testing.assert_array_equal(prepared.tensors['image'], legacy.tensors['image'])
        np.testing.assert_array_equal(prepared.tensors['texts'], legacy.tensors['texts'])
        self.assertEqual(prepared.context, legacy.context)

        raw = task.infer(prepared.tensors)
        self.assertIs(raw['image_feature'], image.raw)
        self.assertIs(raw['text_features'], text.raw)

        result = task.postprocess(raw)
        composed = task.predict(pic, prompts)
        np.testing.assert_array_equal(result.scores, composed.scores)
        np.testing.assert_array_equal(result.order, composed.order)

    def test_predict_routes_through_canonical_stages(self):
        task, _, _, _ = fixture()
        calls = {'pre': 0, 'inf': 0, 'post': 0}
        original = (task.preprocess, task.infer, task.postprocess)

        def counting_preprocess(image, texts):
            calls['pre'] += 1
            return original[0](image, texts)

        def counting_infer(prepared):
            calls['inf'] += 1
            return original[1](prepared)

        def counting_postprocess(outputs):
            calls['post'] += 1
            return original[2](outputs)

        task.preprocess = counting_preprocess
        task.infer = counting_infer
        task.postprocess = counting_postprocess
        task.predict(np.zeros((12, 34, 3), np.uint8), ['a dog'])
        self.assertEqual(calls, {'pre': 1, 'inf': 1, 'post': 1})


class ThinEntryTests(unittest.TestCase):
    def test_main_reexports_local_cli_parser(self):
        from samples.vision.clip.runtime.python import cli, main
        self.assertIs(main.build_parser, cli.build_parser)

    def test_execution_constructs_task_and_calls_predict(self):
        from samples.vision.clip.runtime.python import main, model_runner
        original = model_runner.RuntimeModelRunner
        image_runtime, text_session = ImageRuntime(), TextSession()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            imodel, tmodel = base/'vision.bin', base/'text.onnx'
            imodel.write_bytes(b'host-only fixture')
            tmodel.write_bytes(b'host-only fixture')
            stream = io.StringIO()
            with patch('samples._shared.platforms.detect_target', return_value='x5'), \
                 patch.object(model_runner, 'RuntimeModelRunner',
                              lambda selection: original(selection, image_runtime=image_runtime,
                                                         text_session=text_session)), \
                 patch.object(main, 'save_annotated_image') as save, \
                 contextlib.redirect_stdout(stream):
                rc = main.main(['--target', 'x5', '--image-asset-id', 'x5:clip:img_encoder.bin',
                                '--text-asset-id', 'x5:clip:text_encoder.onnx',
                                '--image-model-path', str(imodel), '--text-model-path', str(tmodel),
                                '--texts', 'a diagram,a dog'])
            self.assertEqual(rc, 0)
            save.assert_called_once()
            report = json.loads(stream.getvalue())
            self.assertEqual(report['prompts'], ['a diagram', 'a dog'])
            self.assertEqual(len(image_runtime.calls), 1)
            self.assertEqual(len(text_session.calls), 1)


if __name__ == '__main__':
    unittest.main()
