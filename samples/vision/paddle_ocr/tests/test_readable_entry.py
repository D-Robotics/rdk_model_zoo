"""Readable-entry checks: thin main, local cli surface, injected execution.

These tests pin the entry structure required by the all-sample readable
runtime design: ``main.py`` re-exports the parser from the local ``cli``
module and visibly constructs :class:`OCRPipeline` before calling
``predict`` — verified here by injecting stage runners and asserting the
exact detection → crop → recognition call pattern, including the
zero-detection short-circuit.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


def _detector_mask(value: float = 0.0) -> np.ndarray:
    return np.full((1, 1, 640, 640), value, dtype=np.float32)


def _rec_scores() -> np.ndarray:
    scores = np.full((1, 40, 97, 1), -3.0, dtype=np.float32)
    scores[:, :, 1, :] = 3.0
    return scores


class ReadableEntryTests(unittest.TestCase):
    def test_main_reexports_local_cli_parser(self):
        from samples.vision.paddle_ocr.runtime.python import cli, main

        self.assertIs(main.build_parser, cli.build_parser)
        # The CLI surface stays a local module; the entry stays thin.
        self.assertTrue(hasattr(cli, "run_list_models"))
        self.assertTrue(hasattr(cli, "run_dry_run"))
        self.assertTrue(hasattr(cli, "run_prepare"))

    def test_execution_constructs_pipeline_and_predicts_with_injected_runners(self):
        from samples.vision.paddle_ocr.runtime.python import main, model_runner

        detector_calls: list = []
        recognizer_calls: list = []

        def detector(inputs):
            detector_calls.append(inputs)
            return {"sigmoid_0.tmp_0": _detector_mask()}

        def recognizer(inputs):
            recognizer_calls.append(inputs)
            return {"softmax_2.tmp_0": _rec_scores()}

        image = np.zeros((32, 48, 3), dtype=np.uint8)
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, patch(
            "utils.py_utils.platforms.require_execution_target", return_value=None
        ), patch.object(
            model_runner,
            "create_stage_runners",
            return_value=(detector, recognizer),
        ) as create, patch.object(
            main, "read_bgr_image", return_value=image
        ), contextlib.redirect_stdout(stream):
            det_model = Path(directory) / "det.bin"
            rec_model = Path(directory) / "rec.bin"
            det_model.write_bytes(b"host-only fixture")
            rec_model.write_bytes(b"host-only fixture")
            arguments = [
                "--target", "x5",
                "--det-asset-id", "x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin",
                "--rec-asset-id", "x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin",
                "--det-model-path", str(det_model),
                "--rec-model-path", str(rec_model),
                "--output-format", "json",
            ]
            self.assertEqual(main.main(arguments), 0)

        create.assert_called_once()
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs.get("priority"), 0)
        self.assertEqual(kwargs.get("bpu_cores"), [0])
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["target"], "x5")
        self.assertEqual(payload["image_shape"], [32, 48, 3])
        # One detector pass, zero boxes, so recognition is skipped entirely.
        self.assertEqual(len(detector_calls), 1)
        self.assertEqual(recognizer_calls, [])
        self.assertEqual(payload["boxes"], [])
        self.assertEqual(payload["texts"], [])


if __name__ == "__main__":
    unittest.main()
