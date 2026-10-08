# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""2026-10-08 runtime-code boundary tests: the pipeline owns model construction.

``OCRPipeline.from_models`` builds both lazy stage runners from a
resolved pair, so ``main`` visibly constructs the pipeline and calls
``predict`` without creating runners or calling load itself.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np


class SimplifiedRuntimeTests(unittest.TestCase):

    def _fake_runtime(self, contract):
        class FakeRuntime:
            model_names = [contract.model_name]
            input_names = {contract.model_name: list(contract.input_names)}
            input_shapes = {contract.model_name: dict(contract.input_shapes)}
            input_dtypes = {contract.model_name: dict(contract.input_dtypes)}
            output_names = {contract.model_name: [contract.output_name]}
            output_shapes = {contract.model_name: {contract.output_name: contract.output_shape}}
            output_dtypes = {contract.model_name: {contract.output_name: "float32"}}

            def run(self, inputs):
                return {contract.model_name: {
                    contract.output_name: np.zeros(contract.output_shape, np.float32)}}

        return FakeRuntime()

    def test_from_models_loads_only_stages_that_execute(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
        from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline, OCRResult

        pair = resolve_pair("x5")
        pipeline = OCRPipeline.from_models(
            pair,
            detector_runtime=self._fake_runtime(pair.detector),
            recognizer_runtime=self._fake_runtime(pair.recognizer),
        )
        self.assertFalse(pipeline.detector_runner.loaded)
        self.assertFalse(pipeline.recognizer_runner.loaded)
        image = np.zeros((64, 320, 3), np.uint8)
        result = pipeline.predict(image)
        self.assertIsInstance(result, OCRResult)
        self.assertTrue(pipeline.detector_runner.loaded)
        self.assertFalse(pipeline.recognizer_runner.loaded)

    def test_main_never_calls_create_stage_runners_or_load(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "runtime" / "python" / "main.py"
        ).read_text()
        self.assertNotIn("create_stage_runners", source)
        self.assertNotIn(".load()", source)


if __name__ == "__main__":
    unittest.main()
