"""NV12 byte contract shared by classification, detection and OCR."""
import subprocess
import sys
import unittest

import numpy as np

from samples._shared.image import bgr_to_nv12_planes


class NV12Tests(unittest.TestCase):
    def test_known_bgr_colors_preserve_limited_range_and_uv_order(self):
        for bgr, luma, chroma in (
            ((0, 0, 0), 16, (128, 128)),
            ((255, 255, 255), 235, (128, 128)),
            ((0, 0, 255), 82, (90, 240)),
            ((0, 255, 0), 145, (54, 34)),
            ((255, 0, 0), 41, (240, 110)),
        ):
            with self.subTest(bgr=bgr):
                y, uv = bgr_to_nv12_planes(np.full((2, 2, 3), bgr, np.uint8))
                np.testing.assert_array_equal(y, np.full((1, 2, 2, 1), luma, np.uint8))
                np.testing.assert_array_equal(uv, np.array(chroma, np.uint8).reshape(1, 1, 1, 2))

    def test_noncontiguous_image_matches_contiguous_pixels(self):
        image = np.random.default_rng(42).integers(0, 256, (12, 20, 3), dtype=np.uint8)
        view = image[::2, ::2]
        for actual, expected in zip(bgr_to_nv12_planes(view), bgr_to_nv12_planes(view.copy())):
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.uint8)
            self.assertTrue(actual.flags.c_contiguous)

    def test_import_does_not_load_image_or_board_dependencies(self):
        result = subprocess.run([sys.executable, '-c',
            "import sys; import samples._shared.image; "
            "assert not {'cv2', 'numpy', 'hbm_runtime', 'hobot_dnn'} & sys.modules.keys()"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
