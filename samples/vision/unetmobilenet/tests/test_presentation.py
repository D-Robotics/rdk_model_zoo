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
main = importlib.import_module('samples.vision.unetmobilenet.runtime.python.main')
for argv in (['--help'], ['--list-models'], ['--dry-run', '--target', 's100']):
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

    def test_nonzero_overlay_keeps_source_colors_and_original_weight(self):
        import cv2
        cli = importlib.import_module('samples.vision.unetmobilenet.runtime.python.cli')
        self.assertTrue(callable(getattr(cli, 'render_overlay', None)), 'CLI owns overlay presentation')
        image = np.arange(3 * 7 * 3, dtype=np.uint8).reshape(3, 7, 3)
        labels = np.full((3, 7), 5, dtype=np.int32)
        colors = np.full(image.shape, (10, 249, 72), dtype=np.uint8)
        expected = cv2.addWeighted(image, 0.75, colors, 0.25, 0.0)
        np.testing.assert_array_equal(cli.render_overlay(image, labels), expected)


if __name__ == '__main__':
    unittest.main()
