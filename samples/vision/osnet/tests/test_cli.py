import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from samples.vision.osnet.runtime.python.cli import build_parser, read_image
from samples.vision.osnet.runtime.python.main import main


class CliTests(unittest.TestCase):
    def test_default_and_explicit_selection_are_sdk_free(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.target, "x5")
        self.assertEqual(args.variant, "spatial-rgb-late8")
        self.assertIsNone(args.test_img)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["--dry-run", "--variant", "spatial-rgb-int16"]), 0)
        self.assertEqual(json.loads(output.getvalue())["variant"], "spatial-rgb-int16")

    def test_reads_owned_rgb_crop_from_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "images.dat"
            path.write_bytes(struct.pack("<III", 1, 2, 1) + bytes([255, 0, 0, 0, 128, 255]))
            image = read_image(fixture_path=path)
            path.unlink()
            np.testing.assert_array_equal(np.asarray(image), [[[255, 0, 0], [0, 128, 255]]])

    def test_rejects_truncated_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "images.dat"
            for data in (b"", struct.pack("<III", 1, 2, 1) + b"x"):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    read_image(fixture_path=path)


if __name__ == "__main__":
    unittest.main()
