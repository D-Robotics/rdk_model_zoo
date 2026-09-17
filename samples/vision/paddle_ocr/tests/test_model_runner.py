"""Host tests for the lazy, metadata-bound OCR stage runner."""

from __future__ import annotations

import unittest

import numpy as np


def _runtime_for_x5_detector(output=None):
    from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair

    pair = resolve_pair("x5")
    contract = pair.detector
    output = (
        np.zeros(contract.output_shape, dtype=np.float32)
        if output is None
        else output
    )

    class FakeRuntime:
        model_names = [contract.model_name]
        input_names = {contract.model_name: list(contract.input_names)}
        input_shapes = {contract.model_name: dict(contract.input_shapes)}
        input_dtypes = {contract.model_name: {"x": "nv12"}}
        output_names = {contract.model_name: [contract.output_name]}
        output_shapes = {contract.model_name: {contract.output_name: contract.output_shape}}
        output_dtypes = {
            contract.model_name: {contract.output_name: "float32"}
        }

        def __init__(self):
            self.calls = []
            self.scheduling = []

        def set_scheduling_params(self, **kwargs):
            self.scheduling.append(kwargs)

        def run(self, inputs):
            self.calls.append(inputs)
            return {contract.model_name: {contract.output_name: output}}

    return pair, FakeRuntime


class ModelRunnerTests(unittest.TestCase):
    def test_injected_runtime_is_lazy_and_returns_flat_output(self):
        from samples.vision.paddle_ocr.runtime.python.model_runner import (
            RuntimeStageRunner,
        )

        pair, runtime_type = _runtime_for_x5_detector()
        runtime = runtime_type()
        runner = RuntimeStageRunner(pair, "detector", runtime=runtime)
        self.assertFalse(runner.loaded)
        output = runner({"x": np.zeros((1, 960, 640, 1), dtype=np.uint8)})
        self.assertTrue(runner.loaded)
        self.assertEqual(tuple(output), ("sigmoid_0.tmp_0",))
        self.assertEqual(runtime.calls[0][pair.detector.model_name].keys(), {"x"})

    def test_scheduling_is_applied_after_lazy_load(self):
        from samples.vision.paddle_ocr.runtime.python.model_runner import (
            RuntimeStageRunner,
        )

        pair, runtime_type = _runtime_for_x5_detector()
        runtime = runtime_type()
        runner = RuntimeStageRunner(
            pair, "detector", runtime=runtime, priority=7, bpu_cores=[0, 1]
        )
        self.assertEqual(runtime.scheduling, [])
        runner({"x": np.zeros((1, 960, 640, 1), dtype=np.uint8)})
        self.assertEqual(
            runtime.scheduling,
            [
                {
                    "priority": {pair.detector.model_name: 7},
                    "bpu_cores": {pair.detector.model_name: [0, 1]},
                }
            ],
        )

    def test_factory_is_not_called_until_execution_and_bad_output_is_rejected(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import (
            MetadataMismatchError,
        )
        from samples.vision.paddle_ocr.runtime.python.model_runner import (
            RuntimeStageRunner,
        )

        pair, runtime_type = _runtime_for_x5_detector(
            np.zeros((1, 1, 640, 639), dtype=np.float32)
        )
        created = []

        def factory(path):
            created.append(path)
            return runtime_type()

        runner = RuntimeStageRunner(pair, "detector", runtime_factory=factory)
        self.assertEqual(created, [])
        with self.assertRaises(MetadataMismatchError):
            runner({"x": np.zeros((1, 960, 640, 1), dtype=np.uint8)})
        self.assertEqual(len(created), 1)

    def test_create_stage_runners_keeps_both_callables_injectable(self):
        from samples.vision.paddle_ocr.runtime.python.model_runner import (
            RuntimeStageRunner,
            create_stage_runners,
        )

        pair, runtime_type = _runtime_for_x5_detector()
        detector, recognizer = create_stage_runners(
            pair,
            priority=0,
            bpu_cores=[0],
            detector_runtime=runtime_type(),
            recognizer_runtime=runtime_type(),
        )
        self.assertIsInstance(detector, RuntimeStageRunner)
        self.assertIsInstance(recognizer, RuntimeStageRunner)
        self.assertEqual(detector.stage, "detector")
        self.assertEqual(recognizer.stage, "recognizer")


if __name__ == "__main__":
    unittest.main()
