# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Synthetic checks for conversion/scripts/board_compare.py (NumPy-only).

The strictness contract of the board-acceptance comparison is behavioral:
exact output-name pairing (never glob order), all requested samples present
on both sides, shape equality, finiteness, declared primary-output roles
(the gate applies per sample and per PRIMARY output — auxiliary outputs
such as audio2mouth's ``feat`` are measured and reported, never gated),
and no averaging across outputs.  These tests pin that behavior with small
synthetic dumps and an injected FP32 double, so they run on any host with
NumPy — no onnxruntime and no board.
"""

import importlib.util
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "conversion"
    / "scripts"
    / "board_compare.py"
)

_spec = importlib.util.spec_from_file_location("lite_avatar_board_compare", SCRIPT)
board_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(board_compare)


def _write_inputs(root, inputs, board, board_prefix="calib"):
    """Create a single-input calib dir and a board dump dir from lists."""

    calib = root / board_prefix
    calib.mkdir(parents=True)
    for i, array in enumerate(inputs):
        np.save(calib / f"{i:03d}.npy", array)
    dump = root / "board_out"
    dump.mkdir(parents=True)
    for i, outputs in enumerate(board):
        for name, array in outputs.items():
            np.save(dump / f"{i:03d}_{name}.npy", array)
    return calib, dump


class BoardCompareStrictnessTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _inputs(self, count=3):
        rng = np.random.default_rng(7)
        return [rng.standard_normal((2, 4)).astype(np.float32) for _ in range(count)]

    def _run_outputs(self, feed):
        x = feed["in0"]
        return [x + 0.5, x * 0.5]

    def test_matching_dump_passes_with_per_sample_per_output_gate(self):
        inputs = self._inputs()
        board = [
            {"out_a": x + 0.5 + 0.01, "out_b": x * 0.5 - 0.01} for x in inputs
        ]
        calib, dump = _write_inputs(self.root, inputs, board)
        samples = board_compare.load_calib(str(calib), ["in0"], False, len(inputs))
        per_out = board_compare.evaluate(
            self._run_outputs, ["in0"], ["out_a", "out_b"], samples, str(dump)
        )
        self.assertEqual(sorted(per_out), ["out_a", "out_b"])
        self.assertEqual([len(v) for v in per_out.values()], [3, 3])
        self.assertTrue(all(v >= 0.99 for v in per_out["out_a"]))
        with redirect_stdout(StringIO()):
            self.assertTrue(board_compare.report(per_out, 0.95, ("out_a", "out_b")))

    def test_missing_board_index_aborts_instead_of_partial_average(self):
        inputs = self._inputs()
        board = [
            {"out_a": x + 0.5, "out_b": x * 0.5} for x in inputs[:2]
        ]  # index 2 never dumped
        calib, dump = _write_inputs(self.root, inputs, board)
        samples = board_compare.load_calib(str(calib), ["in0"], False, 3)
        with self.assertRaises(board_compare.CompareError) as ctx:
            board_compare.evaluate(
                self._run_outputs, ["in0"], ["out_a", "out_b"], samples, str(dump)
            )
        self.assertIn("index 002 missing", str(ctx.exception))

    def test_output_name_difference_aborts_with_both_name_sets(self):
        inputs = self._inputs(1)
        board = [{"out_a": inputs[0] + 0.5, "aux_other": inputs[0]}]
        calib, dump = _write_inputs(self.root, inputs, board)
        samples = board_compare.load_calib(str(calib), ["in0"], False, 1)
        with self.assertRaises(board_compare.CompareError) as ctx:
            board_compare.evaluate(
                self._run_outputs, ["in0"], ["out_a", "out_b"], samples, str(dump)
            )
        message = str(ctx.exception)
        self.assertIn("missing=['out_b']", message)
        self.assertIn("unexpected=['aux_other']", message)

    def test_shape_mismatch_and_nonfinite_board_output_abort(self):
        inputs = self._inputs(1)
        shaped = [{"out_a": np.zeros((2, 5), np.float32), "out_b": inputs[0]}]
        calib, dump = _write_inputs(self.root, inputs, shaped)
        samples = board_compare.load_calib(str(calib), ["in0"], False, 1)
        with self.assertRaises(board_compare.CompareError) as ctx:
            board_compare.evaluate(
                self._run_outputs, ["in0"], ["out_a", "out_b"], samples, str(dump)
            )
        self.assertIn("shape mismatch", str(ctx.exception))

        nonfinite = [
            {"out_a": inputs[0] + 0.5, "out_b": np.full((2, 4), np.nan, np.float32)}
        ]
        calib2, dump2 = _write_inputs(
            self.root / "case2", inputs, nonfinite, board_prefix="calib2"
        )
        samples2 = board_compare.load_calib(str(calib2), ["in0"], False, 1)
        with self.assertRaises(board_compare.CompareError) as ctx:
            board_compare.evaluate(
                self._run_outputs, ["in0"], ["out_a", "out_b"], samples2, str(dump2)
            )
        self.assertIn("board output 'out_b' sample 000", str(ctx.exception))

    def test_calib_short_of_n_is_an_error_not_a_truncation(self):
        inputs = self._inputs(2)
        calib, _ = _write_inputs(self.root, inputs, [])
        with self.assertRaises(board_compare.CompareError) as ctx:
            board_compare.load_calib(str(calib), ["in0"], False, 3)
        self.assertIn("2 sample(s) < requested n=3", str(ctx.exception))

    def test_one_low_sample_fails_the_gate_without_averaging(self):
        # Both outputs primary: out_a always >= gate, out_b dips on one
        # sample — the per-sample minimum fails the gate without averaging.
        inputs = self._inputs(3)
        good = [x * 0.5 for x in inputs]
        # cosine is scale-invariant, so the dip must change direction:
        good[1] = np.random.default_rng(99)  # an unrelated vector on sample 1
        good[1] = good[1].standard_normal(good[0].shape).astype(np.float32)
        board = [
            {"out_a": x + 0.5, "out_b": b} for x, b in zip(inputs, good)
        ]
        calib, dump = _write_inputs(self.root, inputs, board)
        samples = board_compare.load_calib(str(calib), ["in0"], False, 3)
        per_out = board_compare.evaluate(
            self._run_outputs, ["in0"], ["out_a", "out_b"], samples, str(dump)
        )
        self.assertGreaterEqual(np.mean(per_out["out_b"]), 0.0)
        with redirect_stdout(StringIO()) as captured:
            self.assertFalse(board_compare.report(per_out, 0.95, ("out_a", "out_b")))
        self.assertIn("BELOW GATE", captured.getvalue())

    def test_low_auxiliary_output_is_reported_not_gated(self):
        # The acceptance gate is per PRIMARY output: out_b dips hard (a
        # quantization-drifted auxiliary like audio2mouth's feat), out_a
        # passes — the run passes, out_b's numbers stay visible and marked
        # not-gated instead of inventing a false delivery failure.
        inputs = self._inputs(3)
        good = [x * 0.5 for x in inputs]
        good[1] = np.random.default_rng(99)
        good[1] = good[1].standard_normal(good[0].shape).astype(np.float32)
        board = [
            {"out_a": x + 0.5, "out_b": b} for x, b in zip(inputs, good)
        ]
        calib, dump = _write_inputs(self.root, inputs, board)
        samples = board_compare.load_calib(str(calib), ["in0"], False, 3)
        per_out = board_compare.evaluate(
            self._run_outputs, ["in0"], ["out_a", "out_b"], samples, str(dump)
        )
        self.assertLess(np.min(per_out["out_b"]), 0.95)
        with redirect_stdout(StringIO()) as captured:
            self.assertTrue(board_compare.report(per_out, 0.95, ("out_a",)))
        text = captured.getvalue()
        self.assertIn("auxiliary: measured, not gated", text)
        self.assertIn("out_b: n=3", text)  # statistics still reported
        self.assertNotIn("BELOW GATE", text)

    def test_nonfinite_auxiliary_output_still_aborts(self):
        # Roles only steer the gate: strictness (names, shapes, finiteness)
        # applies to every compared output, auxiliary included.
        inputs = self._inputs(1)
        nonfinite = [
            {"out_a": inputs[0] + 0.5, "out_b": np.full((2, 4), np.inf, np.float32)}
        ]
        calib, dump = _write_inputs(self.root, inputs, nonfinite)
        samples = board_compare.load_calib(str(calib), ["in0"], False, 1)
        with self.assertRaises(board_compare.CompareError) as ctx:
            board_compare.evaluate(
                self._run_outputs, ["in0"], ["out_a", "out_b"], samples, str(dump)
            )
        self.assertIn("board output 'out_b' sample 000", str(ctx.exception))

    def test_primary_roles_must_be_declared_and_valid(self):
        resolve = board_compare.resolve_primary_outputs
        outputs = ["output", "feat"]
        # neither mechanism given -> the gate cannot be guessed
        with self.assertRaises(board_compare.CompareError) as ctx:
            resolve(outputs)
        self.assertIn("declared primary outputs", str(ctx.exception))
        # both mechanisms at once is ambiguous
        with self.assertRaises(board_compare.CompareError):
            resolve(outputs, stage="audio2mouth", primary_arg="output")
        # explicit names must exist on the compared graph
        with self.assertRaises(board_compare.CompareError) as ctx:
            resolve(outputs, primary_arg="output,nope")
        self.assertIn("not exposed by the compared graph", str(ctx.exception))
        # known stage contract: audio2mouth gates `output`, never `feat`
        self.assertEqual(resolve(outputs, stage="audio2mouth"), ("output",))
        # a stage whose primaries are absent from the graph is a mismatch
        with self.assertRaises(board_compare.CompareError) as ctx:
            resolve(outputs, stage="audio_encoder")
        self.assertIn("does not expose ['hidden_states']", str(ctx.exception))
        with self.assertRaises(board_compare.CompareError):
            resolve(outputs, stage="unknown_stage")


class _SessionMeta:
    def __init__(self, name):
        self.name = name


class _OrtModuleDouble:
    """Fake ``onnxruntime`` module handing main() a session double.

    ``builds`` records whether a session was constructed, so tests can
    prove --n validation happens before any session work.  The session's
    ``run(names, feed)`` returns values in the REQUESTED names' order,
    computed by name (onnxruntime semantics).
    """

    def __init__(self, inputs, outputs, fn):
        self.builds = 0
        self._inputs = [_SessionMeta(n) for n in inputs]
        self._outputs = [_SessionMeta(n) for n in outputs]
        self._fn = fn

    def InferenceSession(self, path, providers=None):
        self.builds += 1
        return self

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        return [self._fn(name, feed) for name in output_names]


class BoardCompareMainWiringTests(unittest.TestCase):
    """main()/session wiring: exit codes through the real main() path.

    The board side pairs by name-keyed dump files, so these tests pin the
    exit-code contract: pass 0, below-gate 1 (the only path to 1), and
    operational errors — nonpositive --n, a failing report — exit 2 with
    no session work when --n is rejected up front.
    """

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(7)
        self.inputs = [rng.standard_normal((2, 4)).astype(np.float32) for _ in range(3)]

    def _fn(self, name, feed):
        x = feed["in0"]
        return x + 0.5 if name == "out_a" else x * 0.5

    def _write_calib(self):
        calib = self.root / "calib"
        calib.mkdir()
        for i, array in enumerate(self.inputs):
            np.save(calib / f"{i:03d}.npy", array)
        return calib

    def _write_dump(self, boards):
        dump = self.root / "board_out"
        dump.mkdir()
        for i, outputs in enumerate(boards):
            for name, array in outputs.items():
                np.save(dump / f"{i:03d}_{name}.npy", array)
        return dump

    def _run_main(self, argv, boards):
        calib = self._write_calib()
        dump = self._write_dump(boards)
        fake_ort = _OrtModuleDouble(("in0",), ("out_a", "out_b"), self._fn)
        argv = [
            "board_compare.py",
            "--fp32", "fp.onnx",
            "--board_out", str(dump),
            "--calib", str(calib),
            "--n", "3",
            "--gate", "0.95",
        ] + argv
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                with mock.patch("sys.argv", argv), mock.patch.dict(
                    "sys.modules", {"onnxruntime": fake_ort}
                ):
                    board_compare.main()
        return ctx.exception.code, out.getvalue(), err.getvalue(), fake_ort

    def test_matching_dump_passes_exit_0(self):
        boards = [
            {"out_a": x + 0.5, "out_b": x * 0.5} for x in self.inputs
        ]
        code, out, err, _ = self._run_main(["--primary", "out_a,out_b"], boards)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("RESULT: PASS", out)
        self.assertIn("out_a: n=3", out)
        self.assertIn("out_b: n=3", out)

    def test_below_gate_exits_1(self):
        # out_a is an unrelated vector on every sample: the comparison RAN
        # and the primary gate failed -> exit 1, never 2.
        boards = [
            {"out_a": np.flip(x + 0.5), "out_b": x * 0.5} for x in self.inputs
        ]
        code, out, _, _ = self._run_main(["--primary", "out_a,out_b"], boards)
        self.assertEqual(code, 1)
        self.assertIn("RESULT: FAIL", out)
        self.assertIn("BELOW GATE", out)

    def test_nonpositive_n_exits_2_before_any_session(self):
        boards = [
            {"out_a": x + 0.5, "out_b": x * 0.5} for x in self.inputs
        ]
        calib = self._write_calib()
        dump = self._write_dump(boards)
        for bad in ("0", "-2"):
            fake_ort = _OrtModuleDouble(("in0",), ("out_a", "out_b"), self._fn)
            argv = [
                "board_compare.py",
                "--fp32", "fp.onnx",
                "--board_out", str(dump),
                "--calib", str(calib),
                "--primary", "out_a,out_b",
                "--n", bad,
                "--gate", "0.95",
            ]
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    with mock.patch("sys.argv", argv), mock.patch.dict(
                        "sys.modules", {"onnxruntime": fake_ort}
                    ):
                        board_compare.main()
            self.assertEqual(ctx.exception.code, 2, f"n={bad}")
            self.assertIn("must be a positive sample count", err.getvalue())
            self.assertEqual(fake_ort.builds, 0, f"session built for n={bad}")

    def test_report_failure_exits_2_not_1(self):
        real_report = board_compare.report
        self.addCleanup(setattr, board_compare, "report", real_report)
        board_compare.report = mock.Mock(side_effect=RuntimeError("report boom"))
        boards = [
            {"out_a": x + 0.5, "out_b": x * 0.5} for x in self.inputs
        ]
        code, _, err, _ = self._run_main(["--primary", "out_a,out_b"], boards)
        self.assertEqual(code, 2)
        self.assertIn("cannot-run", err)


if __name__ == "__main__":
    unittest.main()
