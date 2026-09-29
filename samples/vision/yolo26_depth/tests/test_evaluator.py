"""Offline depth evaluation contracts, invalid-data rejection and CLI fixture."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from samples.vision.yolo26_depth.evaluator.metrics import DepthMetrics, fidelity_metrics
from samples.vision.yolo26_depth.evaluator.eval_sunrgbd import (
    decode_output,
    load_output,
    main,
)
import test_depth as core


class EvaluatorTests(unittest.TestCase):
    def test_pixel_pooling_and_lower_median(self):
        m = DepthMetrics("none")
        m.update(np.array([[2.0]]), np.array([[1.0]]))
        m.update(np.ones((1, 3)), np.ones((1, 3)))
        r = m.result()
        self.assertEqual(r["valid_pixels"], 4)
        self.assertEqual(r["abs_rel"], 0.25)
        self.assertEqual(r["rmse"], 0.5)
        self.assertEqual(r["delta1"], 0.75)
        # Source torch.median takes the lower middle, unlike np.median.
        median = DepthMetrics("median")
        r = median.update(np.array([[1.0, 3.0]]), np.array([[2.0, 100.0]]))
        self.assertEqual(r["scale"], 2.0)

    def test_no_valid_gt_is_not_perfect_accuracy(self):
        result = DepthMetrics("none").update(np.ones((2, 2)), np.zeros((2, 2)))
        self.assertEqual(result["valid_pixels"], 0)
        self.assertIsNone(result["rmse"])
        with self.assertRaises(ValueError):
            DepthMetrics("unknown")
        with self.assertRaises(ValueError):
            DepthMetrics("none").update(np.ones((2, 2)), np.ones((3, 2)))
        with self.assertRaises(ValueError):
            DepthMetrics("none").update(np.full((2, 2), np.nan), np.ones((2, 2)))

    def test_fidelity_does_not_silently_drop_nonfinite_or_misaligned_data(self):
        r = fidelity_metrics([np.ones((2, 2))], [np.ones((2, 2))])
        self.assertEqual(r["cosine_similarity"], 1.0)
        self.assertEqual(r["max_abs"], 0.0)
        self.assertIsNone(
            fidelity_metrics([np.zeros((2, 2))], [np.zeros((2, 2))])[
                "cosine_similarity"
            ]
        )
        for a, b in (
            ([], []),
            ([np.ones((2, 2))], [np.ones((4,))]),
            ([np.array([np.nan])], [np.array([1.0])]),
        ):
            with self.assertRaises(ValueError):
                fidelity_metrics(a, b)

    def test_decoder_matches_canonical_deployment_profiles(self):
        fixture = core.DepthTests()
        fixture.setUp()
        record = {"original_hw": [37, 23]}
        for target, variant, protocol, boundary in [
            ("x5", "n", "deployment_letterbox", "log"),
            ("s100", "s", "deployment_letterbox", "log"),
            ("s100", "l", "deployment_scale_fill", "raw"),
            ("s600", "x", "deployment_scale_fill", "raw"),
        ]:
            task = fixture.task(target, variant)
            expected = task.predict(fixture.image).depth_native
            actual = decode_output(fixture.raw, record, protocol, boundary, variant)
            np.testing.assert_array_equal(actual, expected)
        with self.assertRaisesRegex(ValueError, "boundary"):
            decode_output(fixture.raw, record, "deployment_letterbox", "raw", "s")
        result = decode_output(
            np.zeros_like(fixture.raw), record, "deployment_letterbox", "log", "s"
        )
        np.testing.assert_array_equal(result, np.ones((37, 23), np.float32))
        self.assertEqual(
            decode_output(
                fixture.raw, record, "ultralytics_validator", "log", "n"
            ).shape,
            (768, 768),
        )

    def test_archive_identity_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.npz"
            for ids in ([0, 0], [0.0, 1.0], [-1, 2]):
                np.savez(
                    p,
                    indices=np.array(ids),
                    float_log=np.zeros((2, 192, 192), np.float32),
                )
                with self.assertRaises(ValueError):
                    load_output(p, "float_log")
            np.savez(
                p,
                indices=np.array([2, 7]),
                float_log=np.zeros((1, 192, 192), np.float32),
            )
            with self.assertRaises(ValueError):
                load_output(p, "float_log")

    def test_preparation_preserves_three_protocols_and_small_screen(self):
        from samples.vision.yolo26_depth.evaluator import prepare_sunrgbd as prep
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            image = np.arange(37 * 23 * 3, dtype=np.uint8).reshape(37, 23, 3)
            cv2.imwrite(str(p / "a.png"), image)
            records = [
                {
                    "index": i,
                    "image": "a.png",
                    "image_hw": [37, 23],
                    "sensor": f"sensor{i}",
                    "depth_m": "gt.npy",
                }
                for i in (2, 7, 11)
            ]
            source = p / "source.json"
            source.write_text(json.dumps({"records": records}))
            with contextlib.redirect_stdout(io.StringIO()):
                prep.main(
                    [
                        "--source-root",
                        str(p),
                        "--source-manifest",
                        str(source),
                        "--output",
                        str(p / "prepared"),
                        "--screen-count",
                        "1",
                    ]
                )
            result = json.loads((p / "prepared/manifest.json").read_text())
            self.assertEqual(len(result["screen_selection"]["indices"]), 1)
            self.assertEqual([r["index"] for r in result["records"]], [2, 7, 11])
            self.assertEqual(
                set(result["protocols"]),
                {
                    "deployment_letterbox",
                    "deployment_scale_fill",
                    "ultralytics_validator",
                },
            )
            r = result["records"][0]
            self.assertEqual(r["ultralytics_validator"]["stage1_hw"], [768, 478])
            self.assertEqual(
                np.load(p / "prepared" / r["deployment_scale_fill"]["npy"]).shape,
                (1, 3, 768, 768),
            )
            self.assertEqual(
                sum(prep.proportional_allocations({"a": [0], "b": [1]}, 0).values()), 0
            )
            self.assertEqual(prep.proportional_allocations({}, 0), {})

    def test_single_image_comparison_has_actual_labels_and_rejects_zero_depth(self):
        from samples.vision.yolo26_depth.evaluator import eval_numeric
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            cv2.imwrite(str(p / "image.png"), np.zeros((20, 30, 3), np.uint8))
            np.save(p / "reference.npy", np.ones((20, 30), np.float32) * 2)
            np.save(p / "candidate.npy", np.ones((20, 30), np.float32))
            args = [
                "--image",
                str(p / "image.png"),
                "--official",
                str(p / "reference.npy"),
                "--candidate",
                str(p / "candidate.npy"),
                "--candidate-name",
                "host-s600-fixture",
                "--output",
                str(p / "out"),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                eval_numeric.main(args)
            report = json.loads((p / "out/comparison-report.json").read_text())
            self.assertEqual(report["candidate"], "host-s600-fixture")
            self.assertEqual(report["median_scale_candidate_to_reference"], 2.0)
            self.assertEqual(report["median_aligned"]["rmse"], 0.0)
            self.assertEqual(
                cv2.imread(str(p / "out/comparison.jpg")).shape, (586, 1620, 3)
            )
            np.save(p / "candidate.npy", np.zeros((20, 30), np.float32))
            with self.assertRaisesRegex(ValueError, "positive"):
                eval_numeric.main([*args[:-1], str(p / "bad")])
            self.assertFalse((p / "bad").exists())

    def test_cli_metric_report_and_missing_ids_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            np.save(p / "gt.npy", np.ones((3, 4), np.float32))
            manifest = {
                "size": 768,
                "protocols": {"deployment_letterbox": {}},
                "records": [
                    {
                        "index": 7,
                        "sample": "fixture",
                        "sensor": "fake",
                        "original_hw": [3, 4],
                        "depth_m": "gt.npy",
                    }
                ],
            }
            (p / "manifest.json").write_text(json.dumps(manifest))
            np.savez(
                p / "float.npz",
                indices=np.array([7]),
                float_log=np.zeros((1, 192, 192), np.float32),
            )
            np.savez(
                p / "quant.npz",
                indices=np.array([7]),
                quant_log=np.zeros((1, 192, 192), np.float32),
            )
            args = [
                "--prepared-manifest",
                str(p / "manifest.json"),
                "--source-root",
                str(p),
                "--float-outputs",
                str(p / "float.npz"),
                "--quant-outputs",
                str(p / "quant.npz"),
                "--candidate-name",
                "host-fixture",
                "--protocol",
                "deployment_letterbox",
                "--boundary",
                "log",
                "--variant",
                "s",
                "--report",
                str(p / "report.json"),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            r = json.loads((p / "report.json").read_text())
            self.assertEqual(r["quant_vs_gt"]["none"]["rmse"], 0.0)
            self.assertEqual(r["board"], "not-run by this offline tool")
            self.assertIsNone(r["quant_vs_float"]["log_depth"]["cosine_similarity"])
            (p / "report.json").unlink()
            np.savez(
                p / "quant.npz",
                indices=np.array([8]),
                quant_log=np.zeros((1, 192, 192), np.float32),
            )
            with self.assertRaisesRegex(ValueError, "indices"):
                main(args)


if __name__ == "__main__":
    unittest.main()
