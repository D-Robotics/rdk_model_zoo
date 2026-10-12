# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Synthetic checks for conversion/scripts/cosine_eval_ptq.py (NumPy-only).

The compiler-side FP32-vs-PTQ comparison must be strict: name sets pair by
identity (never position), all requested samples exist, shapes match, arrays
are finite, primary-output roles are declared (the gate applies per sample
and per PRIMARY output — auxiliary outputs such as audio2mouth's ``feat``
are measured and reported, never gated), and there is no averaging.  These
tests pin that behavior with injected session doubles — no onnxruntime, no
compiled models.
"""

import importlib.util
import json
import os
import tempfile
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
    / "cosine_eval_ptq.py"
)

_spec = importlib.util.spec_from_file_location("lite_avatar_cosine_eval_ptq", SCRIPT)
cosine_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cosine_eval)

INPUTS = ["feats"]
OUTPUTS = ["hidden", "mouth"]


def session_double(transform):
    """A double returning each output run through ``transform``."""

    def run(feed):
        base = feed[INPUTS[0]]
        return [transform(base).copy() for _ in OUTPUTS]

    return run


class CosineEvalPtqStrictnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calib = self.root / "calib"
        self.calib.mkdir()
        rng = np.random.default_rng(7)
        for i in range(3):
            np.save(self.calib / f"{i:03d}.npy", rng.standard_normal((2, 4)).astype(np.float32))

    def test_matching_pair_passes_and_dumps_metrics(self):
        per_out = cosine_eval.evaluate(
            session_double(lambda a: a * 2.0),
            session_double(lambda a: a * 2.0 + 1e-6),
            INPUTS,
            OUTPUTS,
            cosine_eval.load_calib(str(self.calib), INPUTS, False, 3),
        )
        for values in per_out.values():
            self.assertEqual(len(values), 3)
        with redirect_stdout(StringIO()):
            self.assertTrue(cosine_eval.report(per_out, 0.95, tuple(OUTPUTS)))
            dump_dir = self.root / "dump"
            cosine_eval.dump(
                per_out, 0.95, str(dump_dir), "fp.onnx", "pq.onnx", tuple(OUTPUTS)
            )
        import json

        metrics = json.loads((dump_dir / "metrics.json").read_text())
        self.assertTrue(metrics["passed"])
        self.assertEqual(metrics["primary_outputs"], OUTPUTS)
        self.assertEqual(set(metrics["outputs"]), set(OUTPUTS))
        self.assertEqual(
            {role for role in
             (v["role"] for v in metrics["outputs"].values())},
            {"primary"},
        )
        self.assertEqual(metrics["n_samples"], 3)
        raw = np.load(dump_dir / "hidden.cosine.npy")
        self.assertEqual(raw.shape, (3,))

    def test_output_name_set_difference_aborts(self):
        with self.assertRaises(cosine_eval.CompareError) as raised:
            cosine_eval.check_name_sets(
                OUTPUTS, OUTPUTS + ["aux_feat"], "output"
            )
        self.assertIn("unexpected-in-ptq=['aux_feat']", str(raised.exception))

    def test_missing_samples_abort_instead_of_shrinking(self):
        with self.assertRaises(cosine_eval.CompareError) as raised:
            cosine_eval.load_calib(str(self.calib), INPUTS, False, 5)
        self.assertIn("3 sample(s) < requested n=5", str(raised.exception))

    def test_single_bad_sample_fails_gate_without_averaging(self):
        rng = np.random.default_rng(11)

        def ptq_run(feed):
            base = feed[INPUTS[0]]
            outs = [base * 1.5, base * 1.5]
            outs[1][0] = rng.standard_normal(outs[1].shape[1]).astype(np.float32)
            return outs

        samples = cosine_eval.load_calib(str(self.calib), INPUTS, False, 3)
        per_out = cosine_eval.evaluate(
            session_double(lambda a: a * 1.5), ptq_run, INPUTS, OUTPUTS, samples
        )
        with redirect_stdout(StringIO()) as captured:
            self.assertFalse(cosine_eval.report(per_out, 0.95, tuple(OUTPUTS)))
        self.assertIn("BELOW GATE", captured.getvalue())
        # the untouched output must still be reported as ok
        self.assertIn("hidden: n=3", captured.getvalue())

    def test_low_auxiliary_output_is_reported_not_gated(self):
        # A finite-but-low auxiliary (audio2mouth's feat under quantization
        # drift): the primary passes, the auxiliary is reported honestly and
        # creates no false failure; the dump records both roles.
        import json

        rng = np.random.default_rng(11)

        def ptq_run(feed):
            base = feed[INPUTS[0]]
            outs = [base * 1.5, base * 1.5]
            outs[1][0] = rng.standard_normal(outs[1].shape[1]).astype(np.float32)
            return outs

        samples = cosine_eval.load_calib(str(self.calib), INPUTS, False, 3)
        per_out = cosine_eval.evaluate(
            session_double(lambda a: a * 1.5), ptq_run, INPUTS, OUTPUTS, samples
        )
        with redirect_stdout(StringIO()) as captured:
            self.assertTrue(cosine_eval.report(per_out, 0.95, ("hidden",)))
        text = captured.getvalue()
        self.assertNotIn("BELOW GATE", text)
        self.assertIn("auxiliary: measured, not gated", text)
        self.assertIn("mouth: n=3", text)
        dump_dir = self.root / "dump-aux"
        with redirect_stdout(StringIO()):
            cosine_eval.dump(per_out, 0.95, str(dump_dir), "fp.onnx", "pq.onnx", ("hidden",))
        metrics = json.loads((dump_dir / "metrics.json").read_text())
        self.assertTrue(metrics["passed"])
        self.assertEqual(metrics["primary_outputs"], ["hidden"])
        self.assertEqual(metrics["outputs"]["hidden"]["role"], "primary")
        self.assertEqual(metrics["outputs"]["mouth"]["role"], "auxiliary")
        self.assertIsNone(metrics["outputs"]["mouth"]["pass"])

    def test_primary_roles_must_be_declared_and_valid(self):
        resolve = cosine_eval.resolve_primary_outputs
        with self.assertRaises(cosine_eval.CompareError) as ctx:
            resolve(OUTPUTS)
        self.assertIn("declared primary outputs", str(ctx.exception))
        with self.assertRaises(cosine_eval.CompareError):
            resolve(OUTPUTS, stage="audio2mouth", primary_arg="output")
        with self.assertRaises(cosine_eval.CompareError) as ctx:
            resolve(OUTPUTS, primary_arg="hidden,nope")
        self.assertIn("not exposed by the compared graph", str(ctx.exception))
        self.assertEqual(resolve(["output", "feat"], stage="audio2mouth"), ("output",))
        self.assertEqual(resolve(OUTPUTS, primary_arg="hidden"), ("hidden",))

    def test_non_finite_ptq_output_aborts(self):
        samples = cosine_eval.load_calib(str(self.calib), INPUTS, False, 3)

        def bad_run(feed):
            out = feed[INPUTS[0]].copy()
            out[0, 0] = np.inf
            return [out, out]

        with self.assertRaises(cosine_eval.CompareError) as raised:
            cosine_eval.evaluate(
                session_double(lambda a: a), bad_run, INPUTS, OUTPUTS, samples
            )
        self.assertIn("contains non-finite values", str(raised.exception))


class _SessionMeta:
    def __init__(self, name):
        self.name = name


class _OrtSessionDouble:
    """onnxruntime-shaped double for the main()-wiring tests.

    ``run(names, feed)`` returns values in the REQUESTED names' order, each
    computed by name — onnxruntime reorders from the session's own metadata
    order the same way, so a session whose metadata lists outputs in a
    different order still answers any requested order correctly.
    ``output_order`` sets the metadata order ``get_outputs()`` reports.
    """

    def __init__(self, inputs, output_order, fn):
        self._inputs = [_SessionMeta(n) for n in inputs]
        self._outputs = [_SessionMeta(n) for n in output_order]
        self._fn = fn

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        return [self._fn(name, feed) for name in output_names]


class CosineEvalPtqMainWiringTests(unittest.TestCase):
    """main()/session wiring: exit codes and by-name output pairing.

    The PTQ graph may list its outputs in a different metadata order than
    the FP32 graph; the comparison must pair outputs BY NAME through real
    main() wiring.  Output values here are built so that a positional
    mislabel pairs orthogonal vectors and fails the gate: 'output' is a
    constant array, 'feat' is the (random) input itself.
    """

    FP_ORDER = ("output", "feat")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calib = self.root / "calib"
        self.calib.mkdir()
        rng = np.random.default_rng(7)
        self.bases = []
        for i in range(3):
            base = rng.standard_normal((2, 4)).astype(np.float32)
            base[0, 0] = i  # the fns below can tell samples apart by content
            self.bases.append(base)
            np.save(self.calib / f"{i:03d}.npy", base)

        self._fp32_session = None
        self._ptq_session = None
        self._real_makers = (
            cosine_eval.make_fp32_session,
            cosine_eval.make_ptq_session,
        )

        def fake_fp32(path):
            self._fp32_session = _OrtSessionDouble(
                ("input_au",), self.FP_ORDER, self._fp32_fn
            )
            return self._fp32_session

        def fake_ptq(path):
            self._ptq_session = _OrtSessionDouble(
                ("input_au",), self._ptq_order, self._ptq_fn
            )
            return self._ptq_session

        cosine_eval.make_fp32_session = fake_fp32
        cosine_eval.make_ptq_session = fake_ptq
        self.addCleanup(
            self._restore,
            self._real_makers[0],
            self._real_makers[1],
        )

    def _restore(self, real_fp32, real_ptq):
        cosine_eval.make_fp32_session = real_fp32
        cosine_eval.make_ptq_session = real_ptq

    def _fp32_fn(self, name, feed):
        base = feed["input_au"]
        if name == "output":
            return np.ones_like(base)
        return base.copy()

    def _ptq_fn_good(self, name, feed):
        # by-name: 'output' identical to FP32; 'feat' identical except on
        # sample 1, where it becomes an unrelated vector (cosine ~ 0).
        base = feed["input_au"]
        if name == "output":
            return np.ones_like(base)
        value = base.copy()
        if int(base[0, 0]) == 1:
            return np.flip(value).copy()
        return value

    def _run_main(self, argv, ptq_fn, ptq_order=FP_ORDER):
        self._ptq_order = ptq_order
        self._ptq_fn = ptq_fn
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                with mock.patch("sys.argv", argv):
                    cosine_eval.main()
        return ctx.exception.code, out.getvalue(), err.getvalue()

    def _base_argv(self, n="3", **extra):
        argv = [
            "cosine_eval_ptq.py",
            "--fp32", "fp.onnx",
            "--ptq", "pq.onnx",
            "--calib", str(self.calib),
            "--stage", "audio2mouth",
            "--n", str(n),
            "--gate", "0.95",
        ]
        for flag, value in extra.items():
            argv += [flag, value]
        return argv

    def test_reversed_ptq_metadata_pairs_by_name_and_passes(self):
        # PTQ metadata lists the SAME outputs in the REVERSED order; the
        # gate result must come from by-name pairing: pass (exit 0), with
        # 'output' min ~1.0 (primary) and 'feat' min ~0 (auxiliary) in the
        # dumped metrics.  A positional mislabel would pair the constant
        # 'output' with the random 'feat' (cosine ~0) and fail the gate.
        dump = self.root / "dump"
        code, out, err = self._run_main(
            self._base_argv() + ["--dump-dir", str(dump)],
            self._ptq_fn_good,
            ptq_order=("feat", "output"),
        )
        self.assertEqual((code, err), (0, ""))
        self.assertIn("RESULT: PASS", out)
        metrics = json.loads((dump / "metrics.json").read_text())
        self.assertTrue(metrics["passed"])
        self.assertEqual(metrics["primary_outputs"], ["output"])
        out_stats, feat_stats = metrics["outputs"]["output"], metrics["outputs"]["feat"]
        self.assertEqual(out_stats["role"], "primary")
        self.assertEqual(feat_stats["role"], "auxiliary")
        self.assertGreater(out_stats["min"], 0.99)  # by name: untouched
        self.assertLess(feat_stats["min"], 0.5)  # by name: sample-1 flip
        self.assertEqual(out_stats["n"], 3)

    def test_below_gate_by_name_exits_1(self):
        # 'output' itself diverges (an unrelated vector on every sample):
        # the comparison RAN, the primary gate failed -> exit 1, not 2.
        def ptq_fn(name, feed):
            base = feed["input_au"]
            if name == "output":
                return np.flip(base).copy()
            return base.copy()

        code, out, _ = self._run_main(self._base_argv(), ptq_fn)
        self.assertEqual(code, 1)
        self.assertIn("RESULT: FAIL", out)
        self.assertIn("BELOW GATE", out)

    def test_nonpositive_n_exits_2_before_any_session(self):
        for bad in ("0", "-2"):
            code, _, err = self._run_main(self._base_argv(n=bad), self._ptq_fn_good)
            self.assertEqual(code, 2, f"n={bad}")
            self.assertIn("must be a positive sample count", err)
            self.assertIsNone(self._fp32_session, f"session built for n={bad}")
            self.assertIsNone(self._ptq_session, f"session built for n={bad}")

    def test_unwritable_dump_dir_exits_2_not_1(self):
        # Metrics pass but the dump target is unwritable: an operational
        # failure (exit 2), never a gate verdict.
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root bypasses directory permissions")
        ro = self.root / "ro"
        ro.mkdir()
        ro.chmod(0o500)
        self.addCleanup(ro.chmod, 0o700)
        code, _, err = self._run_main(
            self._base_argv() + ["--dump-dir", str(ro / "dump")],
            self._ptq_fn_good,
        )
        self.assertEqual(code, 2)
        self.assertIn("cannot-run", err)

    def test_report_failure_exits_2_not_1(self):
        real_report = cosine_eval.report
        self.addCleanup(setattr, cosine_eval, "report", real_report)
        cosine_eval.report = mock.Mock(side_effect=RuntimeError("report boom"))
        code, _, err = self._run_main(self._base_argv(), self._ptq_fn_good)
        self.assertEqual(code, 2)
        self.assertIn("cannot-run", err)


if __name__ == "__main__":
    unittest.main()
