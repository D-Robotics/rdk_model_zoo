"""Evaluator behavior: streaming comparison, counts, gates, honest PSNR."""

import json
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from samples.vision.lite_avatar.evaluator import evaluate_frames

SAMPLE_DIR = Path(evaluate_frames.__file__).resolve().parents[1]
SCRIPT = SAMPLE_DIR / "evaluator" / "evaluate_frames.py"


def write_frames(directory, frames):
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        cv2.imwrite(str(directory / f"frame_{index:04d}.png"), frame)
    return directory


class EvaluatorMainTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shape = (16, 24, 3)
        self.base = [np.full(shape, value, np.uint8) for value in (60, 100, 140)]

    def _run(self, argv):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = evaluate_frames.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_identical_frames_are_perfect_and_pass_any_gate(self):
        actual = write_frames(self.root / "actual", self.base)
        reference = write_frames(self.root / "ref", self.base)
        code, out, err = self._run(
            ["--actual", str(actual), "--reference", str(reference),
             "--gate", "40"]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("perfect=3", out)
        self.assertIn("mean_psnr_db=None", out)
        self.assertIn("counts_equal=True", out)
        self.assertIn("PASS", out)

    def test_different_frames_report_finite_psnr_and_can_fail_a_gate(self):
        actual = write_frames(self.root / "actual", self.base)
        shifted = [np.clip(frame.astype(np.int16) + 8, 0, 255).astype(np.uint8)
                   for frame in self.base]
        reference = write_frames(self.root / "ref", shifted)
        code, out, _ = self._run(
            ["--actual", str(actual), "--reference", str(reference),
             "--gate", "25"]
        )
        self.assertEqual(code, 0)
        self.assertIn("perfect=0", out)
        # Same +8 shift on every pixel: deterministic PSNR (10*log10(255^2/64)).
        self.assertIn("mean_psnr_db=30.069", out)
        code, _, err = self._run(
            ["--actual", str(actual), "--reference", str(reference),
             "--gate", "45"]
        )
        self.assertEqual(code, 1)
        self.assertIn("FAIL", err)

    def test_truncated_delivery_fails_the_default_full_comparison(self):
        actual = write_frames(self.root / "actual", self.base[:2])
        reference = write_frames(self.root / "ref", self.base)
        code, _, err = self._run(
            ["--actual", str(actual), "--reference", str(reference)]
        )
        self.assertEqual(code, 1)
        self.assertIn("counts differ (actual 2 vs reference 3)", err)
        # An explicit limit is a clearly labeled prefix check and passes.
        code, out, _ = self._run(
            ["--actual", str(actual), "--reference", str(reference),
             "--limit", "2"]
        )
        self.assertEqual(code, 0)
        self.assertIn("prefix_check", out)

    def test_limit_requires_the_advertised_prefix_on_both_sides(self):
        # Both sides shorter than N: the prefix check must not pass by
        # silently comparing fewer frames than advertised.
        both2 = write_frames(self.root / "actual", self.base[:2])
        reference = write_frames(self.root / "ref2", self.base[:2])
        code, _, err = self._run(
            ["--actual", str(both2), "--reference", str(reference),
             "--limit", "3"]
        )
        self.assertEqual(code, 1)
        self.assertIn("needs at least 3 frames on both sides", err)
        self.assertIn("actual 2, reference 2", err)

    def test_empty_or_missing_inputs_are_usage_errors(self):
        reference = write_frames(self.root / "ref", self.base)
        empty = self.root / "empty"
        empty.mkdir()
        for actual in (empty, self.root / "missing"):
            with self.subTest(actual=actual.name):
                code, _, err = self._run(
                    ["--actual", str(actual), "--reference", str(reference)]
                )
                self.assertEqual(code, 2)
                self.assertIn("error:", err)

    def test_invalid_gate_and_limit_are_rejected_before_any_comparison(self):
        actual = write_frames(self.root / "actual", self.base)
        reference = write_frames(self.root / "ref", self.base)
        for argv in (
            ["--actual", str(actual), "--reference", str(reference),
             "--gate", "nan"],
            ["--actual", str(actual), "--reference", str(reference),
             "--gate", "inf"],
            ["--actual", str(actual), "--reference", str(reference),
             "--limit", "0"],
            ["--actual", str(actual), "--reference", str(reference),
             "--limit", "-3"],
        ):
            with self.subTest(argv=argv[-2:]), redirect_stdout(StringIO()):
                with redirect_stderr(StringIO()), self.assertRaises(
                    SystemExit
                ) as raised:
                    evaluate_frames.parse_args(argv)
                self.assertEqual(raised.exception.code, 2)

    def test_undecodable_frame_reports_the_exact_offending_file(self):
        actual = write_frames(self.root / "actual", self.base)
        reference = write_frames(self.root / "ref", self.base)
        # Non-frame files are ignored by the pattern, so the pairing stays
        # aligned; a pattern-matching but undecodable file is the error path.
        (actual / "note.txt").write_text("ignored", encoding="utf-8")
        (reference / "frame_0001.png").write_bytes(b"not an image")
        code, _, err = self._run(
            ["--actual", str(actual), "--reference", str(reference)]
        )
        self.assertEqual(code, 2)
        # The exact offending file name, not a misaligned neighbour.
        self.assertIn("Undecodable frame:", err)
        self.assertIn("frame_0001.png", err)
        self.assertNotIn("frame_0000", err)

    def test_geometry_mismatch_is_an_input_error(self):
        actual = write_frames(self.root / "actual", self.base)
        reference = write_frames(
            self.root / "ref",
            [np.full((20, 24, 3), 60, np.uint8)] + self.base[1:],
        )
        code, _, err = self._run(
            ["--actual", str(actual), "--reference", str(reference)]
        )
        self.assertEqual(code, 2)
        self.assertIn("geometry mismatch", err.lower())

    def test_json_record_is_strict_and_honest_about_perfect_frames(self):
        actual = write_frames(self.root / "actual", self.base)
        reference = write_frames(self.root / "ref", self.base)
        output = self.root / "record.json"
        code, _, _ = self._run(
            ["--actual", str(actual), "--reference", str(reference),
             "--output", str(output)]
        )
        self.assertEqual(code, 0)
        record = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(record["frames_compared"], 3)
        self.assertTrue(record["counts_equal"])
        self.assertFalse(record["prefix_check"])
        self.assertEqual(record["perfect_frames"], 3)
        self.assertIsNone(record["mean_psnr_db"])
        self.assertIsNone(record["min_psnr_db"])
        self.assertEqual(record["mean_mae"], 0.0)
        for item in record["frames"]:
            self.assertIsNone(item["psnr_db"])
            self.assertIn("source", item)

    def test_video_reference_streams_and_matches_equal_frames(self):
        actual = write_frames(self.root / "actual", self.base)
        video = self.root / "ref.mp4"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (24, 16)
        )
        try:
            if not writer.isOpened():
                self.skipTest("cv2 mp4 writer unavailable on this host")
            for frame in self.base:
                writer.write(frame)
        finally:
            writer.release()
        code, out, err = self._run(
            ["--actual", str(actual), "--reference", str(video)]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("counts_equal=True", out)

    def test_script_exit_codes_reach_the_process_level(self):
        actual = write_frames(self.root / "actual", self.base[:2])
        reference = write_frames(self.root / "ref", self.base)
        completed = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--actual", str(actual), "--reference", str(reference)],
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("counts differ", completed.stderr)


if __name__ == "__main__":
    unittest.main()
