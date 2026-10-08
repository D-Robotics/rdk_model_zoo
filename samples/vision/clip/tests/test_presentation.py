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
main = importlib.import_module('samples.vision.clip.runtime.python.main')
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

    def test_nonzero_ranked_annotation_and_lossless_output_match_pixels(self):
        import cv2
        from types import SimpleNamespace
        import tempfile
        cli = importlib.import_module('samples.vision.clip.runtime.python.cli')
        self.assertTrue(callable(getattr(cli, 'draw_scores', None)), 'CLI owns score drawing')
        self.assertTrue(callable(getattr(cli, 'save_image', None)), 'CLI owns image saving')
        image = np.full((120, 900, 3), 50, dtype=np.uint8)
        result = SimpleNamespace(order=np.array([1, 0]), scores=np.array([0.125, 0.875]))
        expected = image.copy()
        cv2.putText(expected, 'Rank 1: dog | similarity: 0.8750', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(expected, 'Rank 2: cat | similarity: 0.1250', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        actual = cli.draw_scores(image, ['cat', 'dog'], result)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.all(image == 50))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'nested', 'result.png')
            cli.save_image(path, actual)
            np.testing.assert_array_equal(cv2.imread(str(path)), expected)


if __name__ == '__main__':
    unittest.main()
