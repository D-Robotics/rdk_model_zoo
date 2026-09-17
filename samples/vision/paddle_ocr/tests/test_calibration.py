"""Host tests for calibration tensor preparation and directory ownership."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


def _load_calibration_module():
    path = Path(__file__).parents[1] / "conversion" / "scripts" / "prepare_calibration.py"
    spec = importlib.util.spec_from_file_location("paddleocr_prepare_calibration", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load calibration helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CalibrationTests(unittest.TestCase):
    def test_tensor_domains_follow_detector_and_recognizer_contracts(self):
        module = _load_calibration_module()
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        image[..., 0] = 255
        detector = module.prepare_tensor(image, target="x5", stage="detector")
        recognizer = module.prepare_tensor(image, target="s100", stage="recognizer")
        self.assertEqual(detector.shape, (1, 3, 640, 640))
        self.assertEqual(detector.dtype, np.float32)
        self.assertEqual(recognizer.shape, (1, 3, 48, 320))
        self.assertEqual(recognizer.dtype, np.float32)
        self.assertAlmostEqual(float(detector[0, 2, 0, 0]), 255.0)
        self.assertAlmostEqual(float(recognizer[0, 2, 0, 0]), 1.0)

    def test_output_directory_rejects_mixed_or_unowned_files(self):
        module = _load_calibration_module()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "calibration"
            output.mkdir()
            (output / "stale.npy").write_bytes(b"stale")
            with self.assertRaises(SystemExit):
                module._prepare_output(output)
            self.assertEqual((output / "stale.npy").read_bytes(), b"stale")

            (output / "stale.npy").unlink()
            module._prepare_output(output)
            (output / "s100_detector_cal_00000.npy").write_bytes(b"tensor")
            with self.assertRaises(SystemExit):
                module._prepare_output(output)
            self.assertEqual(
                (output / "s100_detector_cal_00000.npy").read_bytes(), b"tensor"
            )


if __name__ == "__main__":
    unittest.main()
