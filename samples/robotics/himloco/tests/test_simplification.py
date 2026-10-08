"""Runtime-simplification boundaries for the HIMLoco offline sample.

The CLI/application split is consolidated: argument declarations,
model-free listing/dry-run and the evidence discipline (report reservation,
action-dump records, completion summary) live in ``cli.py``; ``main``
constructs the policy task through the model-owned ``HimLocoTask.from_model``
loader and drives the offline predict loop itself — it never assembles a
runner. The thin ``model_runner`` forwarder and the compatibility
composition ``application.execute`` are gone; the strict
observation-manifest module and the exact X5 publication binding stay.
"""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from samples.robotics.himloco.runtime.python import policy
from samples.robotics.himloco.tests.test_binding import metadata

REPO_ROOT = Path(__file__).resolve().parents[4]

LAZY_SCRIPT = """
import contextlib
import io
import sys

sys.path.insert(0, {root!r})
from samples.robotics.himloco.runtime.python import main

with contextlib.redirect_stdout(io.StringIO()):
    for argv in (["--list-models"], ["--target", "x5", "--dry-run"]):
        assert main.main(argv) == 0, argv
heavy = [
    name
    for name in ("numpy", "scipy", "paddle", "soundfile", "hbm_runtime")
    if name in sys.modules
]
assert not heavy, f"host modes imported heavy modules: {{heavy}}"
"""


class SimplifiedLayoutTests(unittest.TestCase):
    def test_application_and_runner_modules_gone(self):
        package = "samples.robotics.himloco.runtime.python"
        for removed in ("application", "model_runner"):
            self.assertIsNone(
                importlib.util.find_spec(f"{package}.{removed}"),
                f"{removed}.py should be consolidated",
            )

    def test_cli_owns_parser_and_evidence_and_main_reexports_parser(self):
        from samples.robotics.himloco.runtime.python import cli, main

        self.assertIs(main.build_parser, cli.build_parser)
        for name in ("PreparedRun", "prepare", "record_sample", "complete"):
            self.assertTrue(hasattr(cli, name), name)
        self.assertFalse(hasattr(cli, "execute"))
        args = cli.build_parser().parse_args(
            ["--target", "x5", "--input-path", "/tmp/x", "--output-dir", "/tmp/y"]
        )
        self.assertEqual(args.warmup, 10)
        self.assertIsNone(args.priority)

    def test_policy_file_owns_runner_construction(self):
        from samples.robotics.himloco.runtime.python import policy

        self.assertTrue(hasattr(policy, "RuntimeModelRunner"))
        self.assertTrue(hasattr(policy, "HimLocoTask"))

    def test_justified_modules_are_retained(self):
        package = "samples.robotics.himloco.runtime.python"
        for name in ("input_io", "model_binding"):
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


def fake_sdk_runtime(calls=None):
    """A synthetic SDK runtime matching the published policy binding."""
    m = metadata()
    runtime = SimpleNamespace(model_names=m["model_names"])
    for field in (
        "input_names",
        "input_shapes",
        "input_dtypes",
        "output_names",
        "output_shapes",
        "output_dtypes",
    ):
        setattr(runtime, field, {"policy": m[field]})
    calls = calls if calls is not None else []

    def run(feed):
        calls.append(feed)
        return {
            "policy": {"actions": np.arange(1, 13, dtype=np.float32).reshape(1, 12)}
        }

    runtime.run = run
    runtime.set_scheduling_params = Mock()
    return runtime, calls


class ModelOwnedConstructionTests(unittest.TestCase):
    """HimLocoTask.from_model owns runner construction/load and evidence."""

    def test_from_model_loads_and_runs_nonzero_pipeline(self):
        from samples.robotics.himloco.runtime.python.model_binding import (
            resolve_selection,
        )

        runtime, calls = fake_sdk_runtime()
        task = policy.HimLocoTask.from_model(
            resolve_selection("x5"), runtime=runtime
        )
        self.assertIsInstance(task, policy.HimLocoTask)
        # Nonzero observation values run the real packing preprocess, one raw
        # model call and the owned-actions postprocess.
        observation = np.linspace(-1.0, 1.0, 270, dtype=np.float32)
        result = task.predict(observation)
        np.testing.assert_array_equal(
            result.actions, np.arange(1, 13, dtype=np.float32).reshape(1, 12)
        )
        self.assertGreaterEqual(result.latency_ms, 0.0)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(
            calls[0]["policy"]["obs_history"][0], observation.reshape(1, 270)[0]
        )
        self.assertEqual(task.metadata.model_names, ("policy",))
        self.assertIsInstance(task.runtime_module_source, str)
        task.set_scheduling_params(priority=3, bpu_cores=[0])
        runtime.set_scheduling_params.assert_called_once_with(
            priority={"policy": 3}, bpu_cores={"policy": [0]}
        )

    def test_from_model_gates_board_before_sdk_factory(self):
        from samples.robotics.himloco.runtime.python.model_binding import (
            resolve_selection,
        )

        with patch(
            "samples.robotics.himloco.runtime.python.policy.require_execution_target",
            side_effect=ValueError("not x5"),
        ), patch(
            "utils.py_utils.single_array_runner._default_runtime_factory"
        ) as factory:
            with self.assertRaises(ValueError):
                policy.HimLocoTask.from_model(resolve_selection("x5"))
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
