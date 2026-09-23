# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""CLIP entry integration using real task, tokenizer, runner, and local images."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from test_clip import fixture, ImageRuntime, TextSession, SAMPLE


class EntryTests(unittest.TestCase):
    def test_cli_runs_both_encoders_and_saves_annotation(self):
        from samples.vision.clip.runtime.python import main, model_runner
        original = model_runner.RuntimeModelRunner
        image, text = ImageRuntime(), TextSession()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            imodel, tmodel = base/'vision.bin', base/'text.onnx'
            imodel.write_bytes(b'host-only fixture')
            tmodel.write_bytes(b'host-only fixture')
            output = base/'nested/scores.png'
            stream = io.StringIO()
            with patch('samples._shared.platforms.detect_target', return_value='x5'), patch.object(model_runner, 'RuntimeModelRunner', lambda selection: original(selection, image_runtime=image, text_session=text)), contextlib.redirect_stdout(stream):
                rc = main.main(['--target','x5','--image-asset-id','x5:clip:img_encoder.bin',
                                '--text-asset-id','x5:clip:text_encoder.onnx',
                                '--image-model-path',str(imodel),'--text-model-path',str(tmodel),
                                '--img-save-path',str(output),'--texts',' , a diagram ,a dog, '])
            self.assertEqual(rc, 0)
            report = json.loads(stream.getvalue())
            self.assertEqual(report['prompts'], ['a diagram','a dog'])
            self.assertEqual(len(image.calls), 1)
            self.assertEqual(len(text.calls), 1)
            self.assertEqual(image.scheduling, {'priority':{'vision':0},'bpu_cores':{'vision':[0]}})
            saved = cv2.imread(str(output))
            source = cv2.imread(str(SAMPLE/'test_data/dog.jpg'))
            self.assertEqual(saved.shape, source.shape)
            self.assertFalse(np.array_equal(saved, source))

    def test_wrong_board_refuses_before_pipeline(self):
        from samples.vision.clip.runtime.python import main
        with patch('samples._shared.platforms.detect_target', return_value='s100'), patch.object(main, '_run') as run, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main.main(['--target','x5']), 2)
            run.assert_not_called()

    def test_download_fetches_both_manifest_assets_without_network(self):
        from samples.vision.clip.model import download
        with tempfile.TemporaryDirectory() as directory, patch.object(download,'download_asset',return_value='0'*64) as fetch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(download.main(['--target','x5','--output-dir',directory]), 0)
            self.assertEqual({c.args[0].reference for c in fetch.call_args_list},
                             {'x5:clip:img_encoder.bin','x5:clip:text_encoder.onnx'})
            for call in fetch.call_args_list:
                self.assertEqual(call.args[1],Path(directory)/call.args[0].filename)

    def test_invalid_values_and_changed_native_outputs_are_rejected(self):
        task, runner, image, text = fixture()
        for bad in [np.empty((0,10,3),np.uint8),np.zeros((10,10),np.uint8),np.zeros((10,10,3),np.float32)]:
            with self.assertRaises(ValueError): task.pre_process(bad,['a dog'])
        prepared = task.pre_process(np.zeros((10,10,3),np.uint8),['a dog'])
        for values in [np.full((1,3,224,224),np.nan,np.float32),np.full((1,3,224,224),2,np.float32)]:
            with self.assertRaises(ValueError):runner({'image':values,'texts':prepared.tensors['texts']})
        self.assertFalse(image.calls)
        image.raw=np.zeros((1,768),np.float32)
        with self.assertRaises(ValueError):runner(prepared.tensors)
        self.assertFalse(text.calls)
        with self.assertRaises(ValueError): task.post_process({'image_feature':np.full((1,512),np.nan,np.float32),'text_features':np.ones((1,512),np.float32)})

    def test_zero_feature_norm_preserves_source_zero_scores(self):
        task, _, _, _ = fixture()
        result=task.post_process({'image_feature':np.zeros((1,512),np.float32),'text_features':np.zeros((2,512),np.float32)})
        np.testing.assert_array_equal(result.scores,np.zeros(2,np.float32))
        np.testing.assert_array_equal(result.order,np.argsort(result.scores)[::-1])

    def test_encoder_failure_retains_stage_and_does_not_run_later_stage(self):
        _, runner, image, text = fixture()
        tensors = {'image': np.zeros((1,3,224,224),np.float32),
                   'texts': np.zeros((2,77),np.int32)}
        with patch.object(image, 'run', side_effect=RuntimeError('fixture failure')):
            with self.assertRaisesRegex(RuntimeError, 'image encoder stage failed'):
                runner(tensors)
        self.assertFalse(text.calls)
        with patch.object(text, 'run', side_effect=RuntimeError('fixture failure')):
            with self.assertRaisesRegex(RuntimeError, 'text encoder stage failed'):
                runner(tensors)
