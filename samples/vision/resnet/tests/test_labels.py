"""Label file compatibility checks without executing file contents."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest


class LabelTests(unittest.TestCase):
    def test_existing_imagenet_literal_dict_keeps_class_270_label(self):
        from samples.vision.resnet.runtime.python.main import _load_labels

        root = Path(__file__).parents[4]
        labels = _load_labels(
            root / "datasets" / "imagenet" / "imagenet_classes.names"
        )
        self.assertEqual(labels[270], "white wolf, Arctic wolf, Canis lupus tundrarum")

    def test_plain_lines_remain_supported(self):
        from samples.vision.resnet.runtime.python.main import _load_labels

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.txt"
            path.write_text("first\n\nsecond\n", encoding="utf-8")
            self.assertEqual(_load_labels(path), {0: "first", 1: "second"})

    def test_malformed_literal_is_reported(self):
        from samples.vision.resnet.runtime.python.main import _load_labels

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.names"
            path.write_text("{not valid}", encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_labels(path)


if __name__ == "__main__":
    unittest.main()
