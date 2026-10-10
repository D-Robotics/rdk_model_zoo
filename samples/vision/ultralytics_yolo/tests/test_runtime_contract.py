"""Exercise real preprocess methods against simulated runtime metadata."""
from pathlib import Path
import importlib, importlib.util, sys, types, unittest
from unittest.mock import patch, MagicMock
import numpy as np

S = Path(__file__).resolve().parents[1]
from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform


class RuntimeContract(unittest.TestCase):
    def test_segmentation_chw_and_hwc_prototypes_agree(self):
        from test_segmentation_binding import fixture
        hwc, _, _ = fixture(chw=False, quantized=False)
        chw, _, _ = fixture(chw=True, quantized=False)
        left = hwc.post_process(hwc.forward({}), 64, 64)
        right = chw.post_process(chw.forward({}), 64, 64)
        for a, b in zip(left[:3], right[:3]):
            np.testing.assert_allclose(a, b)
        self.assertEqual(len(left[3]), len(right[3]))
        for a, b in zip(left[3], right[3]):
            np.testing.assert_array_equal(a, b)

    def test_every_task_binds_platform_input(self):
        for platform in ('x5', 's100', 's100p', 's600'):
            profile = resolve_platform(platform)
            for module, name, count in [('classify', 'YoloCls', 1),
                                         ('detect', 'YoloDetect', 6),
                                         ('segment', 'YoloSeg', 10),
                                         ('pose', 'YoloPose', 9),
                                         ('detect', 'YoloV10Detect', 6)]:
                if platform == 'x5' and name == 'YoloV10Detect':
                    continue
                with self.subTest(platform=platform, task=module):
                    names = ['image'] if profile.is_packed_input else ['y', 'uv']
                    shapes = {'image': (1, 3, 640, 640)} if profile.is_packed_input else {
                        'y': (1, 640, 640, 1), 'uv': (1, 320, 320, 2)}
                    runtime = types.SimpleNamespace(model_names=['m'], input_names={'m': names},
                        input_shapes={'m': shapes}, output_names={'m': [str(i) for i in range(count)]})
                    fake = types.SimpleNamespace(HB_HBMRuntime=lambda _: runtime)
                    m = importlib.import_module('samples.vision.ultralytics_yolo.runtime.python.' + module)
                    cfg = getattr(m, name + 'Config')(model_path='stub', platform=profile)
                    if name in ('YoloDetect', 'YoloSeg', 'YoloPose', 'YoloCls', 'YoloV10Detect'):
                        # The new detector requires the descriptors provided
                        # by the real SDK, not the former names-only fixture.
                        runtime.input_dtypes = {'m': {n: np.dtype(np.uint8) for n in names}}
                        runtime.output_shapes = {'m': {
                            str(2 * i + j): (1, grid, grid, channels)
                            for i, grid in enumerate((80, 40, 20))
                            for j, channels in enumerate((80, 64))}}
                        runtime.output_dtypes = {'m': {str(i): np.dtype(np.float32) for i in range(count)}}
                        from samples.vision.ultralytics_yolo.runtime.python.backend import build_runner
                        if name == 'YoloCls':
                            from samples.vision.ultralytics_yolo.runtime.python.backend import ClassificationContract, ModelSelection
                            runtime.output_shapes = {'m': {'0': (1,1000,1,1)}}
                            runner = build_runner(ModelSelection('stub',target=platform,task='classify',contract=ClassificationContract()),runtime_loader=lambda:fake)
                        elif name == 'YoloV10Detect':
                            from samples.vision.ultralytics_yolo.runtime.python.backend import DFLDetectionContract, ModelSelection
                            runner=build_runner(ModelSelection('stub',target=platform,contract=DFLDetectionContract(nms='none')),runtime_loader=lambda:fake)
                        elif name == 'YoloSeg':
                            from samples.vision.ultralytics_yolo.runtime.python.backend import DFLSegmentationContract, ModelSelection
                            runtime.output_shapes = {'m': {
                                str(3*i+j): (1, grid, grid, channels)
                                for i, grid in enumerate((80,40,20))
                                for j, channels in enumerate((80,64,32))}}
                            runtime.output_shapes['m']['9'] = (1,160,160,32)
                            selection = ModelSelection('stub',target=platform,task='segment',contract=DFLSegmentationContract())
                            runner = build_runner(selection,runtime_loader=lambda:fake)
                        elif name == 'YoloPose':
                            from samples.vision.ultralytics_yolo.runtime.python.backend import DFLPoseContract, ModelSelection
                            runtime.output_shapes = {'m': {
                                str(3*i+j): (1, grid, grid, channels)
                                for i,grid in enumerate((80,40,20))
                                for j,channels in enumerate((1,64,51))}}
                            selection = ModelSelection('stub',target=platform,task='pose',contract=DFLPoseContract())
                            runner = build_runner(selection,runtime_loader=lambda:fake)
                        else:
                            runner = build_runner(cfg, runtime_loader=lambda: fake)
                        model = getattr(m, name)(cfg, runner=runner)
                    else:
                        with patch('samples.vision.ultralytics_yolo.runtime.python.backend.load_hbm_runtime', return_value=fake):
                            model = getattr(m, name)(cfg)
                    tensors = model.pre_process(np.zeros((24, 36, 3), np.uint8))['m']
                    self.assertEqual(list(tensors), names)
                    self.assertEqual(sum(t.size for t in tensors.values()), 640 * 640 * 3 // 2)
                    for tensor in tensors.values():
                        self.assertEqual(tensor.dtype, np.uint8)

    def test_export_opset_reaches_ultralytics(self):
        p = S / 'conversion/export_monkey_patch.py'
        spec = importlib.util.spec_from_file_location('test_export', p)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fake_model = MagicMock()
        head = types.ModuleType('ultralytics.nn.modules.head')
        for name in ('Detect', 'v10Detect', 'Segment', 'OBB', 'Pose', 'Classify'):
            setattr(head, name, type(name, (), {}))
        block = types.ModuleType('ultralytics.nn.modules.block')
        block.Attention = type('Attention', (), {})
        block.AAttn = type('AAttn', (), {})
        fake_ultralytics = types.ModuleType('ultralytics')
        fake_ultralytics.YOLO = lambda _: fake_model
        modules = {'ultralytics': fake_ultralytics, 'ultralytics.nn.modules.head': head,
                   'ultralytics.nn.modules.block': block, 'torch': types.ModuleType('torch')}
        for flag, value in [('--opset', 19), ('--optse', 11)]:
            with patch.dict(sys.modules, modules), patch.object(sys, 'argv', ['export', flag, str(value)]), patch.object(module, 'modelZooOptimizer'):
                module.main()
            self.assertEqual(fake_model.export.call_args.kwargs['opset'], value)


if __name__ == '__main__':
    unittest.main()
