"""Host regressions for feature-map/stride binding; no BPU SDK required."""
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

SAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAMPLE / 'runtime/python'))
sys.path.insert(0, str(SAMPLE.parents[2]))


class OutputOrderTests(unittest.TestCase):
    def make_runtime(self, task, grid_order=(20, 80, 40), input_size=640):
        channels = {'pose': (1, 4, 51), 'seg': (80, 4, 32), 'obb': (15, 4, 1)}[task]
        shapes = {f'{g}-{c}': (1, g, g, c)
                  for g in grid_order for c in reversed(channels)}
        if task == 'seg':
            shapes = {'prototype': (1, input_size // 4, input_size // 4, 32), **shapes}
        model = types.SimpleNamespace(
            model_names=['m'], input_names={'m': ['image']},
            input_shapes={'m': {'image': (1, 3, input_size, input_size)}},
            output_names={'m': list(shapes)}, output_shapes={'m': shapes})
        sdk = types.ModuleType('hbm_runtime')
        sdk.QuantParams = type('QuantParams', (), {})
        sdk.HB_HBMRuntime = lambda _: model
        previous = sys.modules.get('hbm_runtime')
        sys.modules['hbm_runtime'] = sdk
        try:
            module = importlib.import_module('yolo26_' + task)
        finally:
            if previous is None:
                sys.modules.pop('hbm_runtime', None)
            else:
                sys.modules['hbm_runtime'] = previous
        suffix = {'pose': 'Pose', 'seg': 'Seg', 'obb': 'OBB'}[task]
        config = getattr(module, 'YOLO26' + suffix + 'Config')('unused.bin')
        with patch.object(module, 'hbm_runtime', sdk):
            runtime = getattr(module, 'YOLO26' + suffix)(config)
        return runtime, shapes

    def test_all_tasks_bind_large_maps_to_small_strides(self):
        for task, channels in (('pose', (1, 4, 51)), ('seg', (80, 4, 32)), ('obb', (15, 4, 1))):
            for order in ((20, 80, 40), (80, 40, 20)):
                with self.subTest(task=task, order=order):
                    runtime, _ = self.make_runtime(task, order)
                    expected = [f'{g}-{c}' for g in (80, 40, 20) for c in channels]
                    if task == 'seg':
                        expected.append('prototype')
                    self.assertEqual(runtime.output_names, expected)

    def test_pose_decodes_right_side_person_at_each_stride(self):
        runtime, shapes = self.make_runtime('pose')
        for grid, row, col, box, point in (
            (80, 30, 60, [476, 236, 492, 252], [484, 244]),
            (40, 15, 30, [472, 232, 504, 264], [488, 248]),
            (20, 7, 15, [464, 208, 528, 272], [496, 240]),
        ):
            with self.subTest(grid=grid):
                outputs = {name: np.zeros(shape, np.float32) for name, shape in shapes.items()}
                for g in (20, 40, 80):
                    outputs[f'{g}-1'].fill(-20)
                outputs[f'{grid}-1'][0, row, col, 0] = 3
                outputs[f'{grid}-4'][0, row, col] = 1
                results = runtime.post_process({'m': outputs}, 640, 640)
                self.assertEqual(len(results), 1)
                np.testing.assert_array_equal(results[0]['box'], box)
                np.testing.assert_allclose(results[0]['kpts'][:, :2], [point] * 17)
                self.assertAlmostEqual(results[0]['score'], .95257413, places=6)

    def test_seg_keeps_prototype_after_heads_and_decodes_mask(self):
        runtime, shapes = self.make_runtime('seg')
        outputs = {name: np.zeros(shape, np.float32) for name, shape in shapes.items()}
        for g in (20, 40, 80):
            outputs[f'{g}-80'].fill(-20)
        outputs['20-80'][0, 7, 15, 0] = 3
        outputs['20-4'][0, 7, 15] = 1
        outputs['20-32'][0, 7, 15, 0] = 1
        outputs['prototype'][..., 0] = 10
        boxes, scores, ids, masks = runtime.post_process({'m': outputs}, 640, 640)
        np.testing.assert_allclose(boxes, [[464, 208, 528, 272]])
        self.assertEqual(masks.shape, (1, 640, 640))
        self.assertTrue(masks[0, 240, 496])
        self.assertFalse(masks[0, 10, 10])

    def test_obb_decodes_right_side_center(self):
        runtime, shapes = self.make_runtime('obb')
        outputs = {name: np.zeros(shape, np.float32) for name, shape in shapes.items()}
        for g in (20, 40, 80):
            outputs[f'{g}-15'].fill(-20)
        outputs['20-15'][0, 7, 15, 0] = 3
        outputs['20-4'][0, 7, 15] = 1
        results = runtime.post_process({'m': outputs}, 640, 640)
        self.assertEqual(len(results), 1)
        # OBB26 angle heads emit radians, so a raw zero is horizontal.
        np.testing.assert_allclose(results[0]['rrect'], [496, 240, 64, 64, 0])

    def test_obb_raw_angle_rotates_asymmetric_box_center(self):
        runtime, shapes = self.make_runtime('obb')
        outputs = {name: np.zeros(shape, np.float32) for name, shape in shapes.items()}
        for g in (20, 40, 80):
            outputs[f'{g}-15'].fill(-20)
        outputs['20-15'][0, 7, 15, 9] = 3
        outputs['20-4'][0, 7, 15] = [1, 1, 3, 1]
        outputs['20-1'][0, 7, 15, 0] = np.pi / 2
        result = runtime.post_process({'m': outputs}, 640, 640)[0]
        np.testing.assert_allclose(result['rrect'][:4], [496, 272, 128, 64], atol=1e-4)
        # Orientation is equivalent modulo pi.
        self.assertAlmostEqual(np.cos(2 * result['rrect'][4]), -1, places=6)
        self.assertEqual(result['id'], 9)

    def test_obb_default_labels_follow_ultralytics_dota_ids(self):
        self.make_runtime('obb')
        sdk = sys.modules['yolo26_obb'].hbm_runtime
        previous = sys.modules.get('hbm_runtime')
        sys.modules['hbm_runtime'] = sdk
        try:
            import main
        finally:
            if previous is None:
                sys.modules.pop('hbm_runtime', None)
            else:
                sys.modules['hbm_runtime'] = previous
        from utils.py_utils.file_io import load_class_names
        labels = load_class_names(main.DEFAULT_LABEL_FILES['obb'])
        self.assertEqual(labels[9], 'large-vehicle')
        self.assertEqual(labels[10], 'small-vehicle')
        self.assertEqual(labels[1], 'ship')

    def test_seg_removes_letterbox_padding_at_model_geometry(self):
        for input_size in (64, 640):
            for shape, resize_type in (((input_size, input_size // 2), 1),
                                       ((input_size // 2, input_size), 1),
                                       ((input_size, input_size // 2), 0)):
                with self.subTest(input_size=input_size, shape=shape, resize_type=resize_type):
                    grids = (input_size // 32, input_size // 8, input_size // 16)
                    runtime, shapes = self.make_runtime('seg', grids, input_size)
                    runtime.cfg.resize_type = resize_type
                    outputs = {name: np.zeros(s, np.float32) for name, s in shapes.items()}
                    for g in grids:
                        outputs[f'{g}-80'].fill(-20)
                    grid = input_size // 8
                    row = col = grid // 2
                    outputs[f'{grid}-80'][0, row, col, 0] = 3
                    outputs[f'{grid}-4'][0, row, col] = 1
                    outputs[f'{grid}-32'][0, row, col, 0] = 1
                    outputs['prototype'][..., 0] = 10
                    boxes, _, _, masks = runtime.post_process({'m': outputs}, shape[1], shape[0])
                    self.assertEqual(masks.shape, (1, *shape))
                    x1, y1, x2, y2 = boxes[0].astype(int)
                    # A foreground pixel near the right/bottom interior of the box
                    # must stay aligned after removing either horizontal or vertical padding.
                    self.assertTrue(masks[0, y2 - 3, x2 - 3])
                    self.assertFalse(masks[0, 0, 0])

    def test_seg_predict_resize_override_is_used_for_boxes_and_masks(self):
        runtime, shapes = self.make_runtime('seg', (2, 8, 4), 64)
        outputs = {name: np.zeros(s, np.float32) for name, s in shapes.items()}
        for g in (2, 8, 4):
            outputs[f'{g}-80'].fill(-20)
        outputs['8-80'][0, 4, 4, 0] = 3
        outputs['8-4'][0, 4, 4] = 1
        outputs['8-32'][0, 4, 4, 0] = 1
        outputs['prototype'][..., 0] = 10
        with patch.object(runtime, 'forward', return_value={'m': outputs}):
            boxes, _, _, masks = runtime.predict(np.zeros((64, 32, 3), np.uint8), resize_type=0)
        np.testing.assert_allclose(boxes, [[14, 28, 22, 44]])
        self.assertTrue(masks[0, 40, 20])
        self.assertFalse(masks[0, 40, 26])
        self.assertEqual(runtime.cfg.resize_type, 1)


if __name__ == '__main__':
    unittest.main()
