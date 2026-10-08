"""Runtime-simplification boundaries for the Paraformer sample.

The CLI/application split is consolidated: the per-utterance evidence
helpers and failure records moved from ``application.py`` into ``cli.py``
(whose module surface stays NumPy-free so host listing/dry-run stay light),
and the compatibility loop compositions ``application.run``/``application.execute``
are gone — ``main`` is the single visible composition. The numerical
frontend, CIF bridge, decoder vocabulary and strict manifest modules stay.
"""

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

LAZY_SCRIPT = """
import contextlib
import io
import sys

sys.path.insert(0, {root!r})
from samples.speech.paraformer.runtime.python import main

with contextlib.redirect_stdout(io.StringIO()):
    for argv in (["--list-models"], ["--target", "s100", "--dry-run"]):
        assert main.main(argv) == 0, argv
heavy = [
    name
    for name in ("numpy", "scipy", "torch", "soundfile", "hbm_runtime")
    if name in sys.modules
]
assert not heavy, f"host modes imported heavy modules: {{heavy}}"
"""


class SimplifiedLayoutTests(unittest.TestCase):
    def test_application_module_gone_and_helpers_live_in_cli(self):
        self.assertIsNone(
            importlib.util.find_spec(
                "samples.speech.paraformer.runtime.python.application"
            ),
            "application.py should be consolidated into cli.py",
        )
        from samples.speech.paraformer.runtime.python import cli

        for name in (
            "Preparation",
            "Utterance",
            "prepare",
            "note_runtime",
            "prepare_utterance",
            "mark_attempted",
            "record_prediction",
            "save_features",
            "complete",
            "record_failure",
        ):
            self.assertTrue(hasattr(cli, name), name)

    def test_no_compat_loop_compositions_remain(self):
        from samples.speech.paraformer.runtime.python import cli, main

        for compat in ("run", "execute"):
            self.assertFalse(
                hasattr(cli, compat), f"cli.{compat} compatibility shim remains"
            )
        self.assertIs(main.build_parser, cli.build_parser)

    def test_justified_numerical_modules_are_retained(self):
        package = "samples.speech.paraformer.runtime.python"
        for name in ("frontend", "cif", "decoding", "input_io", "model_binding",
                     "pipeline", "stages", "runtime"):
            self.assertIsNotNone(
                importlib.util.find_spec(f"{package}.{name}"), name
            )

    def test_host_modes_stay_numpy_free_and_lazy(self):
        result = subprocess.run(
            [sys.executable, "-c", LAZY_SCRIPT.format(root=str(REPO_ROOT))],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode, 0, msg=f"stderr: {result.stderr}\n{result.stdout}"
        )


if __name__ == "__main__":
    unittest.main()
