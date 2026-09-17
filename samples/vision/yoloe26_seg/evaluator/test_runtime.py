"""Host regressions for the public YOLOE-26 segmentation contract.

Run with Python, NumPy, OpenCV and SciPy; an HBM/board is not required.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

SAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAMPLE / "runtime" / "python"))
import yoloe26seg as yolo

# Only the unavailable hardware binding is substituted; preprocessing and
# visualization below are the actual repository implementations.
try:
    import hbm_runtime  # noqa: F401
except ImportError:
    sys.modules["hbm_runtime"] = SimpleNamespace(QuantParams=object)
from utils.py_utils.visualize import draw_masks


class RuntimeContractTest(unittest.TestCase):
    """Catch ROI-mask incompatibility and broken staged inference."""

    def test_mask_can_be_rendered_by_shared_visualizer(self):
        """A mask must fit its box, including a non-square letterboxed input."""
        boxes, masks = yolo.restore_masks(
            np.array([[100, 200, 200, 300]], np.float32),
            np.ones((1, 32), np.float32), np.ones((160, 160, 32), np.float32),
            (320, 640, 3),
        )
        np.testing.assert_array_equal(boxes, [[100, 40, 200, 140]])
        self.assertEqual(masks[0].shape, (100, 100))
        self.assertEqual(masks[0].dtype, np.uint8)
        np.testing.assert_array_equal(masks[0], np.ones((100, 100), np.uint8))
        image = np.zeros((320, 640, 3), np.uint8)
        draw_masks(image, boxes, masks, [0], [(100, 50, 20)], alpha=.4)
        np.testing.assert_array_equal(image[40, 100], [40, 20, 8])
        self.assertFalse(image[:40].any())
        self.assertFalse(image[:, :100].any())

    def test_clipped_fractional_and_empty_masks_keep_instance_alignment(self):
        """Clipping and integer ROI conversion must match common draw helpers."""
        boxes, masks = yolo.restore_masks(
            np.array([[-2.5, 1.5, 5.5, 8.5], [700, 1, 710, 9]], np.float32),
            np.ones((2, 32), np.float32), np.ones((160, 160, 32), np.float32),
            (640, 640, 3),
        )
        np.testing.assert_array_equal(boxes, [[0, 1.5, 5.5, 8.5], [640, 1, 640, 9]])
        self.assertEqual([mask.shape for mask in masks], [(7, 5), (8, 0)])
        self.assertFalse(masks[0][0].any())
        self.assertTrue(masks[0][1:].all())

    def test_forward_accepts_prepared_tensors_and_preserves_runtime_result(self):
        """Callers must be able to separate image preparation from inference."""
        model = yolo.YoloE26Seg.__new__(yolo.YoloE26Seg)
        model.model_name = "pf"
        model.input_names = ["y", "uv"]
        raw = {"pf": {"cls_8": np.array([7], np.int32)}}
        model.model = SimpleNamespace(run=Mock(return_value=raw))
        self.assertTrue(callable(getattr(model, "pre_process", None)))
        tensors = model.pre_process(np.zeros((320, 640, 3), np.uint8))
        self.assertEqual(tensors["pf"]["y"].shape, (1, 640, 640, 1))
        self.assertEqual(tensors["pf"]["uv"].shape, (1, 320, 320, 2))
        self.assertIs(model.forward(tensors), raw)
        self.assertIs(model.model.run.call_args.args[0], tensors)
        with self.assertRaises(ValueError):
            model.pre_process(np.zeros((2, 2, 3), np.uint8), image_format="RGB")

    def test_scheduling_with_no_arguments_is_a_noop(self):
        """Omitted scheduling options must not reset runtime scheduling."""
        model = yolo.YoloE26Seg.__new__(yolo.YoloE26Seg)
        model.model_name = "pf"
        model.model = SimpleNamespace(set_scheduling_params=Mock())
        self.assertTrue(callable(getattr(model, "set_scheduling_params", None)))
        model.set_scheduling_params()
        model.model.set_scheduling_params.assert_not_called()
        model.set_scheduling_params(priority=2, bpu_cores=[0])
        model.model.set_scheduling_params.assert_called_once_with(
            priority={"pf": 2}, bpu_cores={"pf": [0]})


if __name__ == "__main__":
    unittest.main()
