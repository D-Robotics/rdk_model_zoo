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
main = importlib.import_module('samples.vision.modnet.runtime.python.main')
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

    def test_nonzero_composite_preserves_matte_endpoints_and_rounding(self):
        cli = importlib.import_module('samples.vision.modnet.runtime.python.cli')
        self.assertTrue(callable(getattr(cli, 'composite', None)), 'CLI owns matte presentation')
        image = np.repeat(np.array([[[20], [100], [240]]], dtype=np.uint8), 3, axis=2)
        background = np.repeat(np.array([[[210], [10], [70]]], dtype=np.uint8), 3, axis=2)
        matte = np.array([[0, 128, 255]], dtype=np.uint8)
        expected = np.repeat(np.array([[[210], [55], [240]]], dtype=np.uint8), 3, axis=2)
        np.testing.assert_array_equal(cli.composite(image, matte, background), expected)


if __name__ == '__main__':
    unittest.main()
