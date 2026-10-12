"""Host tests for board_infer.py helpers (numpy doubles only; no board SDK)."""
import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import numpy as np

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "conversion" / "scripts" / "board_infer.py"


def load_module():
    spec = importlib.util.spec_from_file_location("board_infer", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CollectOutputsTests(unittest.TestCase):
    """No positional fallback: a missing expected name must abort, and the
    returned mapping must pair each name with its own tensor."""

    def setUp(self):
        self.mod = load_module()

    def test_missing_name_aborts_with_actual_keys(self):
        with self.assertRaises(SystemExit) as cm:
            self.mod.collect_outputs({"feat": np.zeros((1,))}, ["output", "feat"])
        self.assertIn("missing: ['output']", str(cm.exception))
        self.assertIn("returned keys ['feat']", str(cm.exception))

    def test_no_first_value_substitution(self):
        # 'output' present but different tensor: must map to its own value,
        # never take feat's (or the first) tensor for 'output'.
        a, b = np.ones((2,)), np.full((2,), 7.0)
        got = self.mod.collect_outputs({"feat": a, "output": b}, ["output", "feat"])
        self.assertTrue((got["output"] == b).all())
        self.assertTrue((got["feat"] == a).all())

    def test_all_present_returns_named_mapping(self):
        a, b = np.zeros((3,)), np.ones((3,))
        got = self.mod.collect_outputs({"output": a, "feat": b}, ["output", "feat"])
        self.assertEqual(sorted(got), ["feat", "output"])


class LoadCalibTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_multi_missing_sample_aborts(self):
        with self.assertRaises(SystemExit):
            self.mod.load_calib("no-such-dir", ["f0"], True, 1)

    def test_single_fewer_than_n_aborts(self):
        with tempfile.TemporaryDirectory() as d:
            np.save(f"{d}/000.npy", np.zeros((1,), np.float32))
            with self.assertRaises(SystemExit):
                self.mod.load_calib(d, ["feats"], False, 2)


class MainArgValidationTests(unittest.TestCase):
    """--n validation fires before the board SDK is touched.

    A sentinel hbm_runtime module turns any attribute access (the
    ``from hbm_runtime import HB_HBMRuntime`` line) into an AssertionError,
    so a passing test proves rejection happened first.
    """

    def setUp(self):
        self.mod = load_module()

    class _SentinelModule(types.ModuleType):
        def __getattr__(self, name):
            raise AssertionError("hbm_runtime SDK touched before --n validation")

    def _run(self, n):
        argv = [
            "board_infer.py", "--hbm", "m.hbm", "--calib", "c", "--out", "o",
            "--n", n,
        ]
        with mock.patch.dict(sys.modules, {"hbm_runtime": self._SentinelModule("hbm_runtime")}):
            with mock.patch("sys.argv", argv):
                with self.assertRaises(SystemExit) as cm:
                    self.mod.main()
        return cm.exception

    def test_nonpositive_n_rejected_before_sdk(self):
        for bad in ("0", "-2"):
            exc = self._run(bad)
            self.assertIn("must be a positive sample count", str(exc))
            self.assertNotEqual(exc.code, 0, f"n={bad}")

    def test_missing_n_rejected_before_sdk(self):
        argv = [
            "board_infer.py", "--hbm", "m.hbm", "--calib", "c", "--out", "o",
        ]
        with mock.patch.dict(sys.modules, {"hbm_runtime": self._SentinelModule("hbm_runtime")}):
            with mock.patch("sys.argv", argv):
                with self.assertRaises(SystemExit) as cm:
                    self.mod.main()
        self.assertIn("--n is required", str(cm.exception))
        self.assertNotEqual(cm.exception.code, 0)


class MainSingleSampleFlowTests(unittest.TestCase):
    """Normal n=1 wiring is unchanged: one sample runs through an SDK double
    and lands in the dump dir (host double; no board needed)."""

    def test_single_sample_end_to_end_with_sdk_double(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as d:
            calib = pathlib.Path(d) / "calib"
            calib.mkdir()
            base = np.arange(4, dtype=np.float32)
            np.save(calib / "000.npy", base)
            out = pathlib.Path(d) / "out"

            class _Dtype:
                def __init__(self, name):
                    self.name = name

            class _FakeModel:
                model_names = ["m"]
                input_names = {"m": ["feats"]}
                input_dtypes = {"m": {"feats": _Dtype("F32")}}
                output_names = {"m": ["hidden"]}
                output_dtypes = {"m": {"hidden": _Dtype("F32")}}
                output_quants = {"m": {}}

                def run(self, feed):
                    return {"m": {"hidden": np.asarray(feed) * 2.0}}

            sdk = types.ModuleType("hbm_runtime")
            sdk.HB_HBMRuntime = lambda path: _FakeModel()
            argv = [
                "board_infer.py", "--hbm", "m.hbm", "--calib", str(calib),
                "--n", "1", "--out", str(out),
            ]
            with mock.patch.dict(sys.modules, {"hbm_runtime": sdk}):
                with mock.patch("sys.argv", argv):
                    with redirect_stdout(StringIO()):
                        self.assertIsNone(mod.main())
            dumped = np.load(out / "000_hidden.npy")
            np.testing.assert_allclose(dumped, base * 2.0)


if __name__ == "__main__":
    unittest.main()
