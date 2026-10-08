"""Reusable classification label validation, independent of any sample."""

import unittest
import numpy as np


class LabelValidationTests(unittest.TestCase):
    def test_sequence_mapping_and_unlabelled_models(self):
        from utils.py_utils.labels import validate_labels

        self.assertIsNone(validate_labels(None, 4))
        labels = ["cat", "dog", "bus", "ship"]
        self.assertIs(validate_labels(labels, 4), labels)
        sparse = {np.int64(2): "bus"}
        self.assertIs(validate_labels(sparse, 4), sparse)

    def test_invalid_label_sets_are_rejected(self):
        from utils.py_utils.labels import validate_labels

        for labels in (["only one"], {-1: "bad"}, {4: "bad"}, {"0": "bad"}):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                validate_labels(labels, 4)
        with self.assertRaises(TypeError):
            validate_labels("abcd", 4)


if __name__ == "__main__":
    unittest.main()
