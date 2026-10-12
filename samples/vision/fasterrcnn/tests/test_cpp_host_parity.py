# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unittest wrapper around the C++ host math harness.

Compiles the SDK-free C++ stages together with the ByteTrack core (host
build, no board SDK) and replays the captured board-stage fixtures through
them, comparing every stage against the product Python runtime.  Run
``run_host_checks.py`` directly for the verbose per-check report.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "cpp" / "run_host_checks.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("frcnn_host_checks", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CppHostParityTest(unittest.TestCase):
    """The C++ CPU stages must reproduce the Python runtime on the fixtures."""

    @classmethod
    def setUpClass(cls):
        if shutil.which("c++") is None and shutil.which("clang++") is None \
                and shutil.which("g++") is None:
            raise unittest.SkipTest("no C++ compiler available on this host")
        cls.module = _load_runner()
        cls.failures = None
        cls.report = ""
        import tempfile

        with tempfile.TemporaryDirectory(prefix="frcnn-host-parity-") as tmp:
            import subprocess

            from pathlib import Path as P

            in_dir = P(tmp) / "in"
            out_dir = P(tmp) / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            expected = cls.module.export_inputs(in_dir)
            binary = cls.module.compile_harness(P(tmp))
            proc = subprocess.run([str(binary), str(in_dir), str(out_dir)],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                cls.failures = [f"host_checks exited {proc.returncode}: "
                                f"{proc.stderr.strip() or proc.stdout.strip()}"]
                cls.report = proc.stdout
                return
            cls.failures = cls.module.compare(out_dir, expected)
            cls.report = proc.stdout

    def test_all_host_checks_pass(self):
        self.assertEqual(self.failures, [],
                         "C++ host parity checks failed:\n" +
                         "\n".join(f"  - {f}" for f in (self.failures or [])))

    def test_rpn_proposal_tolerance_is_absolute_and_discriminating(self):
        # The tolerance must be a strict absolute 1e-4 px: 320 vs 319.997
        # (0.003 px) fails outright — no relative term widens it at large
        # coordinates.  Wrong shape and non-finite values fail too; an
        # exact copy passes.  Ordering is enforced positionally by the same
        # element-wise predicate (covered on the fixture by the numerics
        # test that swaps the farthest-apart rows).
        matches = self.module.proposals_match
        expected = np.full((2, 4), 320.0, dtype=np.float32)
        drifted = expected.copy()
        drifted[0, 2] = np.float32(319.997)
        self.assertFalse(matches(drifted, expected))
        self.assertFalse(matches(np.zeros((3, 4), dtype=np.float32), expected))
        nonfinite = expected.copy()
        nonfinite[1, 1] = np.float32("nan")
        self.assertFalse(matches(nonfinite, expected))
        self.assertTrue(matches(expected.copy(), expected))


if __name__ == "__main__":
    unittest.main()
