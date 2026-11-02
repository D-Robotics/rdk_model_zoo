import tempfile
import unittest
from pathlib import Path

import numpy as np

from samples.vision.osnet.evaluator.metrics import records, retrieval


class RetrievalTests(unittest.TestCase):
    def test_excludes_same_camera_and_junk_but_keeps_distractors(self):
        query = [(Path("q"), 7, 1)]
        gallery = [(Path("same-camera"), 7, 1), (Path("junk"), -1, 2),
                   (Path("distractor"), 0, 2), (Path("match"), 7, 2)]
        qf, gf = np.zeros((1, 512), np.float32), np.zeros((4, 512), np.float32)
        gf[:, 0] = [0, .1, .2, .3]
        result = retrieval(query, gallery, qf, gf, normalize=False)
        self.assertEqual(result["rank1_percent"], 0)
        self.assertEqual(result["mAP_percent"], 50)

    def test_normalized_distance_matches_cosine_ordering(self):
        query = [(Path("q"), 1, 1)]
        gallery = [(Path("match"), 1, 2), (Path("other"), 2, 2)]
        qf, gf = np.zeros((1, 512), np.float32), np.zeros((2, 512), np.float32)
        qf[0, 0] = 1
        gf[0, 0] = 10
        gf[1, :2] = [1, 1]
        self.assertEqual(retrieval(query, gallery, qf, gf)["rank1_percent"], 100)
        self.assertEqual(retrieval(query, gallery, qf, gf, normalize=False)["rank1_percent"], 0)

    def test_record_order_filters_junk_and_keeps_pid_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ["0002_c2_a.jpg", "-1_c1_a.jpg", "0000_c1_a.jpg"]:
                (Path(folder) / name).touch()
            self.assertEqual([pid for _, pid, _ in records(folder)], [0, 2])

    def test_rejects_invalid_features_and_empty_ground_truth(self):
        with self.assertRaises(ValueError):
            retrieval([(Path("q"), 1, 1)], [], np.zeros((1, 256)), np.zeros((0, 512)))
        with self.assertRaises(ValueError):
            retrieval([(Path("q"), 1, 1)], [(Path("g"), 2, 1)], np.zeros((1, 512)), np.zeros((1, 512)))


if __name__ == "__main__":
    unittest.main()
