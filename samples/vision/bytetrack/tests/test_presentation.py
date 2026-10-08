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
main = importlib.import_module('samples.vision.bytetrack.runtime.python.main')
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

    def test_nonzero_track_overlay_preserves_id_color_and_input(self):
        from types import SimpleNamespace
        cli = importlib.import_module('samples.vision.bytetrack.runtime.python.cli')
        self.assertTrue(callable(getattr(cli, 'draw_tracks', None)), 'CLI owns track drawing')
        image = np.full((70, 100, 3), 45, dtype=np.uint8)
        before = image.copy()
        track = SimpleNamespace(track_id=3, tlbr=(10.0, 30.0, 60.0, 60.0))
        result = cli.draw_tracks(image, [track])
        np.testing.assert_array_equal(result[45, 10], [111, 51, 87])
        np.testing.assert_array_equal(result[0, 0], [45, 45, 45])
        np.testing.assert_array_equal(image, before)
        self.assertFalse(np.array_equal(result, image))


if __name__ == '__main__':
    unittest.main()
