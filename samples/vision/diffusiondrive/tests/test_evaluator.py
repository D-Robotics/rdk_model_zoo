"""Offline metrics must not accept broadcast shapes or fabricated zero cosine."""

import contextlib, io, json, tempfile, unittest
from pathlib import Path
import numpy as np
from samples.vision.diffusiondrive.evaluator.compare_outputs import (
    compare,
    cosine,
    main,
)
from samples.vision.diffusiondrive.runtime.python.model_binding import (
    bind_model,
    resolve_selection,
)
from samples.vision.diffusiondrive.runtime.python.diffusiondrive import (
    DiffusionDriveTask,
)
from test_diffusiondrive import metadata, arrays, SOURCE


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.reference = arrays("reference_outputs.npz")
        self.decoded = DiffusionDriveTask(
            lambda _: None, bind_model(resolve_selection("s600"), metadata())
        ).post_process(self.reference)

    def test_reference_self_comparison_and_semantic_metrics(self):
        report = compare(self.reference, self.decoded)
        for values in report["tensors"].values():
            self.assertAlmostEqual(values["cosine"], 1)
            self.assertEqual(values["mae"], 0)
        self.assertEqual(report["bev"]["pixel_agreement"], 1)
        self.assertEqual(report["bev"]["mean_iou"], 1)
        self.assertEqual(report["status"], "descriptive; no acceptance threshold")
        self.assertFalse(report["dataset_accuracy"])

    def test_rejects_broadcast_shapes_nonfinite_wrong_labels_and_inconsistent_logits(
        self,
    ):
        for key, value in [
            ("bev_labels", self.decoded["bev_labels"][0]),
            ("trajectory", np.zeros((8, 3), np.float32)),
            ("agent_scores", np.full((1, 30), np.nan, np.float32)),
            ("bev_labels", np.full((1, 128, 256), 7, np.uint8)),
            ("bev_labels", np.zeros((1, 128, 256), np.uint8)),
        ]:
            with self.assertRaises(ValueError):
                compare(self.reference, {**self.decoded, key: value})
        with self.assertRaises(ValueError):
            compare(
                {**self.reference, "trajectory": np.zeros((8, 3), np.float32)},
                self.decoded,
            )

    def test_zero_norm_is_undefined_and_shapes_are_exact(self):
        self.assertIsNone(cosine(np.zeros(3), np.zeros(3)))
        self.assertIsNone(cosine(np.ones(3), np.zeros(3)))
        with self.assertRaises(ValueError):
            cosine(np.ones((1, 3)), np.ones(3))

    def test_cli_records_exact_archive_hashes_and_will_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            np.savez(p / "decoded.npz", **self.decoded)
            args = [
                "--reference-npz",
                str(SOURCE / "test_data/reference_outputs.npz"),
                "--board-npz",
                str(p / "decoded.npz"),
                "--output",
                str(p / "metrics.json"),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            report = json.loads((p / "metrics.json").read_text())
            self.assertEqual(len(report["reference_sha256"]), 64)
            self.assertEqual(len(report["candidate_sha256"]), 64)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 2)


if __name__ == "__main__":
    unittest.main()
