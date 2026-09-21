"""Static consistency checks between the C++ runtime constants and the Python contracts."""
from pathlib import Path
import re, sys, unittest

S = Path(__file__).resolve().parents[1]
R = S.parents[2]
sys.path.insert(0, str(R))
sys.path.insert(0, str(S / 'runtime/python'))

CPP_DETECT = (S / 'runtime/cpp/detect/main.cc').read_text()
CPP_DECODE = (S / 'runtime/cpp/common/decode.h').read_text()
CPP_POSE = (S / 'runtime/cpp/pose/main.cc').read_text()
CPP_SEGMENT = (S / 'runtime/cpp/segment/main.cc').read_text()


class CppContractTests(unittest.TestCase):
    def test_detect_constants_match_python_contract(self):
        from yolo26_det import YOLO26DetectConfig
        from yolo_detect import YoloDetectConfig
        self.assertIn('const int kClasses = 80;', CPP_DETECT)
        self.assertIn('const int kStrides[] = {8, 16, 32};', CPP_DETECT)
        self.assertEqual(YOLO26DetectConfig(model_path='x').classes_num, 80)
        self.assertEqual(YoloDetectConfig(model_path='x').classes_num, 80)
        self.assertEqual(YOLO26DetectConfig(model_path='x').strides, [8, 16, 32])

    def test_decode_primitives_cover_both_head_contracts(self):
        self.assertIn('kDflBins = 16;', CPP_DECODE)
        self.assertIn('decode_box_ltrb', CPP_DECODE)
        self.assertIn('decode_box_dfl', CPP_DECODE)
        self.assertIn('BoxDecode::kDirectLtrb', CPP_DETECT)
        self.assertIn('BoxDecode::kDfl', CPP_DETECT)
        # The channel-count probe must accept both 4- and 64-channel box maps.
        self.assertIn('if (channels == 4) return BoxDecode::kDirectLtrb;', CPP_DECODE)
        self.assertIn('if (channels == 4 * kDflBins) return BoxDecode::kDfl;', CPP_DECODE)

    def test_pose_and_segment_dispatch_on_box_channels(self):
        for source in (CPP_POSE, CPP_SEGMENT):
            with self.subTest(source='pose' if source is CPP_POSE else 'segment'):
                self.assertIn('direct_ltrb', source)
                self.assertIn('offset * (direct_ltrb ? 4 : 4 * REG)', source)

    def test_pose_keypoint_constants(self):
        self.assertIn('#define KPT_NUM 17', CPP_POSE)
        self.assertIn('#define KPT_ENCODE 3', CPP_POSE)

    def test_segment_mask_constants(self):
        self.assertIn('#define MCES 32', CPP_SEGMENT)

    def test_all_tasks_support_both_input_protocols(self):
        for task in ('classify', 'detect', 'pose', 'segment'):
            source = (S / f'runtime/cpp/{task}/main.cc').read_text()
            with self.subTest(task=task):
                if task == 'detect':
                    # Input plumbing lives in common/dnn_io via probe_input_protocol.
                    self.assertIn('probe_input_protocol', source)
                    self.assertIn('Nv12Input', source)
                else:
                    self.assertIn('HB_DNN_IMG_TYPE_NV12', source)
                    self.assertIn('i420_to_packed_nv12', source)
                    self.assertIn('i420_to_split_nv12', source)


if __name__ == '__main__':
    unittest.main()
