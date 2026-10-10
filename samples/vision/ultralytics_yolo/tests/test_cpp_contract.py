"""Static consistency checks between the C++ runtime constants and the Python contracts."""
from pathlib import Path
import re, sys, unittest
import shutil, subprocess, tempfile

S = Path(__file__).resolve().parents[1]
R = S.parents[2]
sys.path.insert(0, str(R))

CPP_DETECT = (S / 'runtime/cpp/detect/main.cc').read_text()
CPP_DECODE = (S / 'runtime/cpp/common/decode.h').read_text()
CPP_DNN_IO = (S / 'runtime/cpp/common/dnn_io.cc').read_text()
CPP_POSE = (S / 'runtime/cpp/pose/main.cc').read_text()
CPP_SEGMENT = (S / 'runtime/cpp/segment/main.cc').read_text()


class CppContractTests(unittest.TestCase):
    def test_detect_constants_match_python_contract(self):
        from samples.vision.ultralytics_yolo.runtime.python.detect import YOLO26DetectConfig
        from samples.vision.ultralytics_yolo.runtime.python.detect import YoloDetectConfig
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
                self.assertIn('direct_ltrb ? 4 : 4 * REG', source)

    def test_pose_keypoint_constants(self):
        self.assertIn('#define KPT_NUM 17', CPP_POSE)
        self.assertIn('#define KPT_ENCODE 3', CPP_POSE)

    def test_segment_mask_constants(self):
        self.assertIn('#define MCES 32', CPP_SEGMENT)

    def test_all_tasks_support_both_input_protocols(self):
        session = (S / 'runtime/cpp/common/task_session.h').read_text()
        for task in ('classify', 'detect', 'pose', 'segment', 'obb'):
            source = (S / f'runtime/cpp/{task}/main.cc').read_text()
            with self.subTest(task=task):
                # Input plumbing lives in common/dnn_io via probe_input_protocol;
                # task programs reach it through the shared TaskSession.
                plumbing = source if task == 'detect' else session
                if task != 'detect':
                    self.assertIn('yolo::TaskSession', source)
                self.assertIn('probe_input_protocol', plumbing)
                self.assertIn('Nv12Input', plumbing)
        # The protocol detection itself is shared, not duplicated per task.
        self.assertIn('HB_DNN_IMG_TYPE_NV12', CPP_DNN_IO)
        # Execute production plumbing against both SDK API doubles. Helper
        # spelling is not a behavioral contract: owned Y/UV can bypass I420.
        compiler = shutil.which('c++')
        if compiler is None:
            self.skipTest('C++ compiler required for native input protocol tests')
        cpp = S / 'runtime/cpp'
        with tempfile.TemporaryDirectory() as directory:
            for stack in ('x5', 'ucp'):
                with self.subTest(stack=stack):
                    binary = Path(directory) / stack
                    command = [compiler, '-std=c++11', '-Wall', '-Wextra', '-Werror',
                               '-I', str(cpp / 'test/fake_dnn_io' / stack),
                               '-I', str(cpp), str(cpp / 'test/test_dnn_io.cc'),
                               str(cpp / 'common/dnn_io.cc'),
                               str(cpp / 'common/nv12_geometry.cc'), '-o', str(binary)]
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
