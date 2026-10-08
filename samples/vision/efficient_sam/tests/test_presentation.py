# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Presentation values and model-free command dependency boundaries."""
import importlib
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[4]


class PresentationTests(unittest.TestCase):
    def test_model_free_modes_do_not_import_image_libraries_or_sdk(self):
        # A fresh interpreter catches both direct and transitive eager imports.
        script = """
import contextlib
import io
import sys
import importlib
from unittest.mock import patch
main = importlib.import_module('samples.vision.efficient_sam.runtime.python.main')
for argv in (['--help'], ['--list-models'], ['--dry-run', '--target', 'x5']):
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            status = main.main(argv)
        except SystemExit as exc:
            status = exc.code
    assert status == 0, (argv, status)
    assert not any(name.split('.')[0] in {'numpy', 'cv2', 'hbm_runtime'} for name in sys.modules), argv
"""
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_nonzero_mask_overlay_preserves_blending_and_input(self):
        import cv2
        cli = importlib.import_module('samples.vision.efficient_sam.runtime.python.cli')
        self.assertTrue(callable(getattr(cli, 'draw_mask_result', None)), 'CLI owns mask drawing')
        image = np.full((31, 57, 3), (83, 91, 107), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=bool)
        mask[150:350, 180:370] = True
        actual = cli.draw_mask_result(image, mask, 0.8123, 1)
        self.assertEqual(actual.shape, (512, 512, 3))
        original = np.full((1, 1, 3), (83, 91, 107), dtype=np.uint8)
        color = np.full((1, 1, 3), (0, 180, 0), dtype=np.uint8)
        expected_inside = cv2.addWeighted(original, 0.45, color, 0.55, 0)[0, 0]
        np.testing.assert_array_equal(actual[240, 240], expected_inside)
        np.testing.assert_array_equal(actual[450, 450], [83, 91, 107])
        np.testing.assert_array_equal(image, np.full((31, 57, 3), (83, 91, 107), dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
