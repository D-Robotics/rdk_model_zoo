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
main = importlib.import_module('samples.vision.pp_liteseg.runtime.python.main')
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

    def test_nonzero_three_panels_keep_geometry_and_car_color(self):
        import cv2
        cli = importlib.import_module('samples.vision.pp_liteseg.runtime.python.cli')
        self.assertTrue(callable(getattr(cli, 'render_result', None)), 'CLI owns result presentation')
        image = np.arange(57 * 91 * 3, dtype=np.uint8).reshape(57, 91, 3)
        labels = np.full((512, 1024), 13, dtype=np.int32)
        result = cli.render_result(image, labels)
        self.assertEqual(result.shape, (548, 3078, 3))
        original = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(result[36:, :1024], original)
        expected_segmentation = np.full((512, 1024, 3), (142, 0, 0), dtype=np.uint8)
        np.testing.assert_array_equal(result[36:, 2054:], expected_segmentation)
        np.testing.assert_array_equal(cli.colorize(labels), expected_segmentation)


if __name__ == '__main__':
    unittest.main()
