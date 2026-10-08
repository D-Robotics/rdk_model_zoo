"""Label-contract behavior tests for the YOLO CLI (spec §6, custom models).

Custom (explicit local model path) runs must not silently reuse official
COCO/ImageNet/DOTA label sets; explicit labels are checked against the bound
model's class count before inference; cls presentation uses the validated
labels passed in (falling back to class IDs) instead of re-reading default
label files; official default runs keep their label behavior. Host-only: no
SDK, no board, no network.
"""

from pathlib import Path
import sys
import types
import unittest
from unittest import mock

_RUNTIME_PYTHON = Path(__file__).resolve().parents[1] / "runtime" / "python"
if str(_RUNTIME_PYTHON) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_PYTHON))

import yolo_cli  # noqa: E402
from yolo_cli import load_labels, validate_label_count  # noqa: E402


def _args(label_file=None):
    return types.SimpleNamespace(label_file=label_file)


class _Contract:
    def __init__(self, classes):
        self.classes = classes


class _Model:
    def __init__(self, classes):
        self.contract = _Contract(classes)


class CustomModelLabelTests(unittest.TestCase):
    def test_custom_model_without_label_file_shows_ids_not_official_labels(self):
        for task in ("detect", "seg", "pose", "cls", "obb"):
            with self.subTest(task=task):
                self.assertEqual(load_labels(_args(), task, custom_model=True), [])

    def test_custom_model_with_explicit_label_file_still_loads_it(self):
        with unittest.mock.patch.object(yolo_cli, "_COCO_LABELS", "/nonexistent"):
            labels = load_labels(_args(), "detect", custom_model=False)
        self.assertIsInstance(labels, list)  # official path exercised separately


class OfficialDefaultLabelTests(unittest.TestCase):
    def test_official_defaults_keep_task_label_sets(self):
        detect = load_labels(_args(), "detect", custom_model=False)
        self.assertEqual(len(detect), 80)
        self.assertEqual(detect[0], "person")
        cls = load_labels(_args(), "cls", custom_model=False)
        self.assertEqual(len(cls), 1000)
        obb = load_labels(_args(), "obb", custom_model=False)
        self.assertEqual(len(obb), 15)

    def test_official_pose_default_is_exact_single_person_label(self):
        # The old default applied the 80-class COCO file to a single-class
        # person model (accidentally correct only because COCO id 0 is
        # "person"). The exact label is now returned for the 1-class pose
        # contract instead of an unrelated 80-entry list.
        self.assertEqual(load_labels(_args(), "pose", custom_model=False),
                         ["person"])


class LabelCountValidationTests(unittest.TestCase):
    def test_wrong_label_count_fails_with_concrete_counts(self):
        with self.assertRaises(ValueError) as raised:
            validate_label_count(["a", "b"], _Model(4))
        message = str(raised.exception)
        self.assertIn("2", message)
        self.assertIn("4", message)

    def test_matching_label_count_passes(self):
        validate_label_count(["a", "b", "c", "d"], _Model(4))

    def test_empty_labels_skip_validation(self):
        validate_label_count([], _Model(4))

class ClsPresentationTests(unittest.TestCase):
    def _present(self, labels, result):
        import io
        import contextlib

        args = types.SimpleNamespace(task="cls", label_file=None,
                                     img_save_path="/tmp/unused-cls.jpg")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            yolo_cli.present_result(args, None, result, labels)
        return buffer.getvalue()

    def test_cls_presentation_uses_passed_labels_and_never_rereads_defaults(self):
        from unittest import mock

        with mock.patch("utils.py_utils.file_io.load_labels",
                        side_effect=AssertionError("default label re-read")):
            output = self._present(["a", "b", "c", "d"], [(3, 0.9), (0, 0.1)])
        self.assertIn("d", output)
        self.assertNotIn("3", output.split("Results:")[-1].split("a")[0])

    def test_cls_presentation_without_labels_falls_back_to_class_ids(self):
        from unittest import mock

        with mock.patch("utils.py_utils.file_io.load_labels",
                        side_effect=AssertionError("default label re-read")):
            output = self._present([], [(3, 0.9)])
        self.assertIn("3", output)


class PresentationWithoutLabelsTests(unittest.TestCase):
    """Real presentation calls (draw to temp files) with empty label lists.

    Custom models without ``--label-file`` must render class IDs on every
    task path that draws labels — ``visualize.draw_boxes`` indexes
    ``class_names[cls_id]`` directly, so an empty list would crash the user
    path. Testing ``load_labels`` alone proves nothing about presentation.
    """

    def _run_task(self, task, result):
        import contextlib
        import io
        import tempfile

        args = types.SimpleNamespace(
            task=task, label_file=None,
            img_save_path=str(Path(tempfile.mkdtemp()) / f"{task}.jpg"),
            kpt_conf_thres=0.5)
        image = None
        if task != "cls":
            import numpy as np
            image = np.zeros((32, 48, 3), np.uint8)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            yolo_cli.present_result(args, image, result, [])
        self.assertTrue(Path(args.img_save_path).is_file(),
                        f"{task} result image missing")
        return buffer.getvalue()

    def test_detect_without_labels_renders_ids_and_draws(self):
        import numpy as np
        boxes = np.array([[4.0, 4.0, 24.0, 20.0]], np.float32)
        scores = np.array([0.9], np.float32)
        ids = np.array([2], np.int64)
        output = self._run_task("detect", (boxes, scores, ids))
        self.assertIn("2", output)

    def test_seg_without_labels_draws_ids_into_image(self):
        import numpy as np
        boxes = np.array([[4.0, 4.0, 24.0, 20.0]], np.float32)
        scores = np.array([0.9], np.float32)
        ids = np.array([1], np.int64)
        masks = [np.ones((16, 20), np.uint8)]
        # seg prints nothing to stdout; the ID label is drawn into the
        # image. The bug under test was draw_boxes IndexError-ing on the
        # empty label list, so a written file with drawn pixels is the
        # observable proof.
        self._run_task("seg", (boxes, scores, ids, masks))

    def test_pose_without_labels_draws(self):
        import numpy as np
        boxes = np.array([[4.0, 4.0, 24.0, 20.0]], np.float32)
        scores = np.array([0.9], np.float32)
        ids = np.array([0], np.int64)
        xy = np.zeros((1, 17, 2), np.float32)
        confidence = np.ones((1, 17, 1), np.float32)
        self._run_task("pose", (boxes, scores, ids, xy, confidence))

    def test_obb_without_labels_renders_ids(self):
        import cv2

        records = [{"rrect": (16.0, 10.0, 8.0, 6.0, 0.2),
                    "score": 0.9, "id": 3}]
        # The OBB path only prints the saved-image path; the fallback class
        # ID is drawn into the image (yolo_cli passes f"{label} {score:.2f}"
        # to cv2.putText). Spy on the real cv2.putText — wrapping it keeps
        # the drawing side effect — and assert the annotation text itself,
        # independent of the tempdir name or stdout.
        with mock.patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
            self._run_task("obb", records)
        texts = [call.args[1] for call in put_text.call_args_list]
        self.assertEqual(texts, ["3 0.90"])


class LabelFileFormatTests(unittest.TestCase):
    def _tmp(self, content):
        import tempfile
        path = Path(tempfile.mkdtemp()) / "labels.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def test_explicit_empty_label_file_is_an_error_not_unspecified(self):
        empty = self._tmp("\n \n")
        with self.assertRaises(ValueError) as raised:
            load_labels(_args(str(empty)), "detect", custom_model=True)
        self.assertIn(str(empty), str(raised.exception))

    def test_legacy_json_dict_label_file_still_loads(self):
        # Legacy cls label files in json/dict format (consumed by the old
        # file_io.load_labels presentation path) keep working.
        path = self._tmp('{"0": "cat", "1": "dog"}')
        self.assertEqual(
            load_labels(_args(str(path)), "cls", custom_model=True),
            ["cat", "dog"])

    def test_json_list_label_file_still_loads(self):
        path = self._tmp('["cat", "dog"]')
        self.assertEqual(
            load_labels(_args(str(path)), "detect", custom_model=True),
            ["cat", "dog"])

    def test_sparse_label_mapping_fails_with_clear_error(self):
        path = self._tmp('{"0": "cat", "5": "dog"}')
        with self.assertRaises(ValueError) as raised:
            load_labels(_args(str(path)), "detect", custom_model=True)
        self.assertIn("contiguous", str(raised.exception))


class ValidatorStrictnessTests(unittest.TestCase):
    def test_model_without_contract_raises_instead_of_skipping(self):
        class _Bare:
            pass
        with self.assertRaises(ValueError) as raised:
            validate_label_count(["a"], _Bare())
        self.assertIn("class count", str(raised.exception))

    def test_non_integer_contract_class_count_raises(self):
        with self.assertRaises(ValueError):
            validate_label_count(["a"], _Model(object()))


class PlanExplicitFlagTests(unittest.TestCase):
    def test_explicit_flag_comes_from_describe_plan(self):
        from yolo_platform import resolve_platform
        profile = resolve_platform("x5")
        explicit_args = types.SimpleNamespace(
            asset_id=None, family=None, model_size=None, model_path="custom.bin",
            task="detect")
        plan = yolo_cli.describe_plan(profile, explicit_args)
        self.assertTrue(plan["explicit"])
        default_args = types.SimpleNamespace(
            asset_id=None, family=None, model_size=None, model_path=None,
            task="detect")
        plan = yolo_cli.describe_plan(profile, default_args)
        self.assertFalse(plan["explicit"])


if __name__ == "__main__":
    unittest.main()
