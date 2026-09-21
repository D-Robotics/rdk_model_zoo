"""Family inference from published asset filenames across every registry family."""
from pathlib import Path
import sys, unittest

S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S / 'runtime/python'))

from yolo_assets import family_from_filename  # noqa: E402
from yolo_platform import resolve_platform  # noqa: E402


class FamilyInferenceTests(unittest.TestCase):
    def test_x5_published_names_map_to_their_families(self):
        profile = resolve_platform('x5')
        cases = {
            'yolo26n_detect_bayese_640x640_nv12.bin': 'yolo26',
            'yolov5un_detect_bayese_640x640_nv12.bin': 'yolov5u',
            'yolov8n_seg_bayese_640x640_nv12.bin': 'yolov8',
            'yolov9t_detect_bayese_640x640_nv12.bin': 'yolov9',
            'yolov10n_detect_bayese_640x640_nv12.bin': 'yolov10',
            'yolo11n_pose_bayese_640x640_nv12.bin': 'yolo11',
            'yolo12n_detect_bayese_640x640_nv12.bin': 'yolo12',
            'yolov13n_detect_bayese_640x640_nv12.bin': 'yolov13',
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(family_from_filename(profile, filename), expected)

    def test_s_series_names_map_to_their_families(self):
        profile = resolve_platform('s600')
        cases = {
            'yolo26n_detect_nashp_640x640_nv12.hbm': 'yolo26',
            'yolo11n_detect_nashp_640x640_nv12.hbm': 'yolo11',
            'yolov10n_detect_nashp_640x640_nv12.hbm': 'yolov10',
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(family_from_filename(profile, filename), expected)

    def test_longest_token_wins(self):
        # yolov10 must not collapse into a shorter registered prefix.
        profile = resolve_platform('x5')
        self.assertEqual(
            family_from_filename(profile, 'yolov10s_detect_bayese_640x640_nv12.bin'),
            'yolov10')

    def test_unknown_filename_returns_none(self):
        profile = resolve_platform('x5')
        self.assertIsNone(
            family_from_filename(profile, 'mobilenet_v2_bayese_224x224_nv12.bin'))
        self.assertIsNone(family_from_filename(profile, ''))


if __name__ == '__main__':
    unittest.main()
