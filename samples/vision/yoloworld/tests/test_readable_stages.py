# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""The readable stage surface (preprocess/infer/postprocess) on YOLOWorldTask.

These tests pin the accepted readable-runtime contract: the new stage names
carry the real implementation, the established ``pre_process``/``forward``/
``post_process`` names delegate to them (no second implementation),
``predict`` equals the explicit three-step chain with this call's prompt
context, one ``predict`` is exactly one runner call, and different prompts
keep their own text-embedding context. All execution uses the injected host
fixture from ``test_yoloworld``; no board SDK is loaded and no board
inference is claimed.
"""

from unittest import mock
import unittest
import numpy as np

from test_yoloworld import fixture


class YoloWorldReadableStageTests(unittest.TestCase):
    def test_new_stage_names_and_predict_equality(self):
        task, runtime = fixture()
        runtime.scores[0, 7, 0] = .9
        runtime.boxes[0, 7] = [1, 2, 100, 200]
        image = np.zeros((20, 50, 3), np.uint8)
        prepared = task.preprocess(image, ['dog'])
        staged = task.postprocess(task.infer(prepared), prepared.context)
        composed = task.predict(image, ['dog'])
        for name in ('boxes', 'scores', 'class_ids'):
            np.testing.assert_array_equal(
                getattr(staged, name), getattr(composed, name))
        self.assertEqual(staged.prompts, composed.prompts)
        self.assertGreater(len(composed.scores), 0)

    def test_legacy_stage_names_delegate_to_the_new_methods(self):
        task, _ = fixture()
        image = np.zeros((20, 50, 3), np.uint8)
        with mock.patch.object(task, 'preprocess', wraps=task.preprocess) as pre, \
                mock.patch.object(task, 'infer', wraps=task.infer) as infer, \
                mock.patch.object(task, 'postprocess', wraps=task.postprocess) as post:
            prepared = task.pre_process(image, ['dog'])
            task.post_process(task.forward(prepared), prepared.context)
        pre.assert_called_once()
        infer.assert_called_once()
        post.assert_called_once()

    def test_predict_executes_the_runner_exactly_once(self):
        task, runtime = fixture()
        before = len(runtime.calls)
        task.predict(np.zeros((37, 59, 3), np.uint8), ['dog'])
        self.assertEqual(len(runtime.calls), before + 1)

    def test_different_prompts_and_sizes_keep_their_own_context(self):
        task, runtime = fixture()
        runtime.scores[0, 7, 0] = .9
        runtime.boxes[0, 7] = [1, 2, 100, 200]
        wide = np.full((200, 300, 3), 30, np.uint8)
        small = np.full((40, 28, 3), 200, np.uint8)
        for image, prompts in ((wide, ['dog']), (small, ['person', 'dog'])):
            prepared = task.preprocess(image, prompts)
            staged = task.postprocess(task.infer(prepared), prepared.context)
            composed = task.predict(image, prompts)
            for name in ('boxes', 'scores', 'class_ids'):
                np.testing.assert_array_equal(
                    getattr(staged, name), getattr(composed, name))
            self.assertEqual(prepared.context.prompts, tuple(prompts))


if __name__ == '__main__':
    unittest.main()
