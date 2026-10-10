"""Static consistency checks between the C++ runtime constants and the Python contracts."""
from pathlib import Path
import re, sys, unittest
import shutil, subprocess, tempfile

S = Path(__file__).resolve().parents[1]
R = S.parents[2]
sys.path.insert(0, str(R))

CPP = S / 'runtime/cpp'
CPP_DETECT = (CPP / 'src/detect.cpp').read_text()
CPP_DECODE = (CPP / 'inc/yolo.hpp').read_text()
CPP_BACKEND = (CPP / 'src/backend.cpp').read_text()
CPP_POSE = (CPP / 'src/pose.cpp').read_text()
CPP_SEGMENT = (CPP / 'src/segment.cpp').read_text()
CPP_OBB = (CPP / 'src/obb.cpp').read_text()


class CppContractTests(unittest.TestCase):
    def test_detect_constants_match_python_contract(self):
        from samples.vision.ultralytics_yolo.runtime.python.detect import YOLO26DetectConfig
        from samples.vision.ultralytics_yolo.runtime.python.detect import YoloDetectConfig
        self.assertIn('const int kClasses = 80;', CPP_DETECT)
        self.assertIn('const int kStrides[3] = {8, 16, 32};', CPP_DETECT)
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
                self.assertIn('offset * (direct_ltrb ? 4 : 4 * 16)', source)

    def test_pose_keypoint_constants(self):
        self.assertIn('const int kKptNum = 17;', CPP_POSE)
        self.assertIn('const int kKptEncode = 3;', CPP_POSE)

    def test_segment_mask_constants(self):
        self.assertIn('const int kMces = 32;', CPP_SEGMENT)

    def test_obb_head_geometry_and_platform_policy(self):
        self.assertIn('struct RotatedBox', CPP_DECODE)
        self.assertIn('inline bool decode_obb_cell', CPP_DECODE)
        self.assertIn('inline void regularize_obb', CPP_DECODE)
        self.assertIn('inline bool map_obb_to_source', CPP_DECODE)
        self.assertIn('struct ObbHeadPlan', CPP_DECODE)
        self.assertIn('inline ObbHeadPlan bind_obb_heads', CPP_DECODE)
        self.assertIn('bind_obb_heads', CPP_OBB)
        self.assertIn('decode_obb_cell', CPP_OBB)
        # X5 wraps angles, runs per-class rotated NMS and clips; the S series
        # keeps class-agnostic NMS with unclipped geometry.
        self.assertIn('classwise_rotated_nms', CPP_OBB)
        self.assertIn('YOLO_DNN_STACK_X5', CPP_OBB)

    def test_all_tasks_support_both_input_protocols(self):
        for task in ('classify', 'detect', 'pose', 'segment', 'obb'):
            source = (CPP / f'src/{task}.cpp').read_text()
            with self.subTest(task=task):
                # Input plumbing lives in the shared backend via
                # probe_input_protocol; every task consumes it the same way.
                self.assertIn('probe_input_protocol', source)
                self.assertIn('Nv12Input', source)
        # The protocol detection itself is shared, not duplicated per task.
        self.assertIn('HB_DNN_IMG_TYPE_NV12', CPP_BACKEND)
        # Execute production plumbing against both SDK API doubles. Helper
        # spelling is not a behavioral contract: owned Y/UV can bypass I420.
        compiler = shutil.which('c++')
        if compiler is None:
            self.skipTest('C++ compiler required for native input protocol tests')
        with tempfile.TemporaryDirectory() as directory:
            for stack in ('x5', 'ucp'):
                with self.subTest(stack=stack):
                    binary = Path(directory) / stack
                    command = [compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                               '-I', str(CPP / 'test/fake_dnn_io' / stack),
                               '-I', str(CPP / 'inc'), str(CPP / 'test/test_dnn_io.cc'),
                               str(CPP / 'src/backend.cpp'), '-o', str(binary)]
                    subprocess.run(command, check=True, capture_output=True,
                                   text=True, timeout=120)
                    subprocess.run([str(binary)], check=True, capture_output=True,
                                   text=True, timeout=120)


if __name__ == '__main__':
    unittest.main()
