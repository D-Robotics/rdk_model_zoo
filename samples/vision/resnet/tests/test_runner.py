"""Host tests for the injected runner seam and lazy SDK boundary."""

from __future__ import annotations

import unittest

import numpy as np


class _FakeRuntime:
    model_names = ["resnet18_224x224_nv12"]
    input_names = {"resnet18_224x224_nv12": ["data"]}
    input_shapes = {"resnet18_224x224_nv12": {"data": (1, 3, 224, 224)}}
    input_dtypes = {"resnet18_224x224_nv12": {"data": "U8"}}
    output_names = {"resnet18_224x224_nv12": ["prob"]}
    output_shapes = {
        "resnet18_224x224_nv12": {"prob": (1, 1000, 1, 1)}
    }
    output_dtypes = {"resnet18_224x224_nv12": {"prob": "F32"}}

    def __init__(self):
        self.calls = []

    def run(self, payload):
        self.calls.append(payload)
        return {
            "resnet18_224x224_nv12": {
                "prob": np.zeros((1, 1000, 1, 1), dtype=np.float32)
            }
        }


class RunnerTests(unittest.TestCase):
    def test_injected_runtime_loads_without_board_sdk_or_hardware(self):
        from samples.vision.resnet.runtime.python.model_binding import resolve_selection
        from samples.vision.resnet.runtime.python.model_runner import RuntimeModelRunner

        fake = _FakeRuntime()
        selection = resolve_selection("x5")
        runner = RuntimeModelRunner(selection, runtime=fake)
        binding = runner.load()

        output = runner(
            {binding.input_names[0]: np.zeros((1, 336, 224, 1), dtype=np.uint8)}
        )
        self.assertEqual(tuple(output[binding.output_name].shape), (1, 1000, 1, 1))
        self.assertEqual(fake.calls[0][binding.model_name][binding.input_names[0]].dtype, np.uint8)

    def test_runtime_factory_is_called_only_on_load(self):
        from samples.vision.resnet.runtime.python.model_binding import resolve_selection
        from samples.vision.resnet.runtime.python.model_runner import RuntimeModelRunner

        created = []

        def factory(path):
            created.append(path)
            return _FakeRuntime()

        selection = resolve_selection("x5")
        runner = RuntimeModelRunner(selection, runtime_factory=factory)
        self.assertEqual(created, [])
        runner.load()
        self.assertEqual(created, [str(selection.model_path)])


if __name__ == "__main__":
    unittest.main()
