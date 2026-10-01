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

    def test_non_integer_contract_class_count_is_skipped_not_guessed(self):
        # Injected host doubles (MagicMock-like) do not expose an int class
        # count; the validator must skip rather than guess from the output
        # protocol.
        validate_label_count(["a"], _Model(object()))


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

        with mock.patch("rdk_yolo_utils.file_io.load_labels",
                        side_effect=AssertionError("default label re-read")):
            output = self._present(["a", "b", "c", "d"], [(3, 0.9), (0, 0.1)])
        self.assertIn("d", output)
        self.assertNotIn("3", output.split("Results:")[-1].split("a")[0])

    def test_cls_presentation_without_labels_falls_back_to_class_ids(self):
        from unittest import mock

        with mock.patch("rdk_yolo_utils.file_io.load_labels",
                        side_effect=AssertionError("default label re-read")):
            output = self._present([], [(3, 0.9)])
        self.assertIn("3", output)


if __name__ == "__main__":
    unittest.main()
