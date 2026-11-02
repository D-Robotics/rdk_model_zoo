"""Calibration tensors retain raw RGB values and reject incomplete image sets."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'conversion/prepare_calibration.py'
SPEC = importlib.util.spec_from_file_location('convnext_calibration', SCRIPT)
producer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(producer)


class CalibrationTests(unittest.TestCase):
    def test_raw_rgb_nchw_bytes_and_incomplete_set_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / 'images'
            images.mkdir()
            Image.new('RGB', (500, 300), (17, 83, 201)).save(images / 'one.png')
            with self.assertRaises(ValueError):
                producer.prepare(images, root / 'incomplete', count=2)
            self.assertFalse((root / 'incomplete').exists())
            result = producer.prepare(images, root / 'tensors', count=1)
            tensor = np.fromfile(root / 'tensors/0000.rgb', dtype=np.float32).reshape(3, 224, 224)
            self.assertEqual(result['count'], 1)
            for channel, value in enumerate((17, 83, 201)):
                np.testing.assert_array_equal(tensor[channel], value)
            with self.assertRaises(FileExistsError):
                producer.prepare(images, root / 'tensors', count=1)
