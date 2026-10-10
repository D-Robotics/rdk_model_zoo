"""Runtime-simplification boundaries for the KWS sample.

The consolidated layout splits the former monolithic entry into a thin
``main.py`` (construct the task through ``KWS.from_model``, apply
scheduling, call ``predict`` once) plus a ``cli.py`` that owns option
declarations, model-free listing/dry-run and the probability report. The
model file ``kws.py`` owns the runner construction, the model-owned loader
``KWS.from_model`` and the real postprocess scoring, so the thin
``model_runner``/``postprocess`` forwarders are gone and ``main`` never
assembles a runner. Model listing stays NumPy-free; dry-run (which
validates the fixed frontend Config) must stay free of
SciPy/Paddle/SoundFile/SDK.
"""

import contextlib
import importlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from samples.speech.kws.runtime.python import kws as kws_module
from samples.speech.kws.runtime.python import cli as audio_io
from samples.speech.kws.runtime.python import kws as frontend
from samples.speech.kws.runtime.python.cli import resolve_selection
from samples.speech.kws.runtime.python.kws import bind_model
from samples.speech.kws.tests.test_runner import FakeRuntime

REPO_ROOT = Path(__file__).resolve().parents[4]

LAZY_SCRIPT = """
import contextlib
import io
import sys

sys.path.insert(0, {root!r})
from samples.speech.kws.runtime.python import main

with contextlib.redirect_stdout(io.StringIO()):
    # Model listing stays NumPy-free (the parser/binding chain has no NumPy).
    assert main.main(["--list-models"]) == 0
    assert "numpy" not in sys.modules, "list-models imported NumPy"
with contextlib.redirect_stdout(io.StringIO()):
    # Dry-run validates the fixed frontend Config before printing, which
    # imports the NumPy-based frontend module; everything heavier must stay
    # out (established behavior, preserved).
    assert main.main(["--target", "s100", "--dry-run"]) == 0
heavy = [
    name
    for name in ("scipy", "paddle", "soundfile", "hbm_runtime")
    if name in sys.modules
]
assert not heavy, f"dry-run imported heavy modules: {{heavy}}"
"""


def metadata(dtype="float32", quant=None):
    return dict(
        model_name="kws",
        model_names=["kws"],
        input_names=["features"],
        output_names=["score"],
        input_shapes={"features": [1, 373, 80]},
        output_shapes={"score": [1, 2, 1]},
        input_dtypes={"features": "float32"},
        output_dtypes={"score": dtype},
        output_quants={"score": quant} if quant else {},
    )


class SimplifiedLayoutTests(unittest.TestCase):
    def test_model_file_owns_runner_and_scoring(self):
        self.assertTrue(hasattr(kws_module, "RuntimeModelRunner"))
        for removed in ("model_runner", "postprocess"):
            self.assertIsNone(
                importlib.util.find_spec(
                    f"samples.speech.kws.runtime.python.{removed}"
                ),
                f"{removed}.py should no longer exist as a forwarder module",
            )

    def test_cli_owns_options_and_report_and_main_reexports_parser(self):
        from samples.speech.kws.runtime.python import cli, main

        self.assertIs(main.build_parser, cli.build_parser)
        for name in ("run_list_models", "run_dry_run", "build_report",
                     "write_report"):
            self.assertTrue(hasattr(cli, name), name)
        parser = cli.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.audio_maxlen, 60000)
        self.assertEqual(args.threshold, 0.5)
        self.assertEqual(args.bpu_cores, [0])
        self.assertFalse(args.list_models)
        self.assertFalse(args.dry_run)

    def test_stage_postprocess_is_real_scoring_not_forwarding(self):
        binding = bind_model(
            resolve_selection("s100"),
            metadata(
                "int8",
                SimpleNamespace(
                    scale=[0.01], zero_point=[0], axis=0, quant_type="SCALE"
                ),
            ),
        )
        task = kws_module.KWS(lambda tensors: None, binding)
        # Dequantization must happen before the max reduction.
        self.assertAlmostEqual(
            task.postprocess(np.array([[[10], [90]]], np.int8)), 0.9, places=6
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


class ModelOwnedConstructionTests(unittest.TestCase):
    """KWS.from_model owns runner construction/load; main stays high-level."""

    def test_from_model_loads_and_runs_nonzero_pipeline(self):
        config = kws_module.Config()
        fake = FakeRuntime()
        task = kws_module.KWS.from_model(
            resolve_selection("s100"),
            config,
            frontend=lambda waveform, cfg: np.full((373, 80), 0.25, np.float32),
            runtime=fake,
        )
        self.assertIsInstance(task, kws_module.KWS)
        # Nonzero 16 kHz waveform: padding to 60000 samples and fbank feature
        # extraction run before the single raw model call.
        waveform = np.linspace(-1, 1, 40000, dtype=np.float32)
        self.assertAlmostEqual(task.predict(waveform, 16000), 0.985, places=6)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["kws"]["features"].shape, (1, 373, 80))
        self.assertEqual(task.metadata.output_names, ("score",))
        task.set_scheduling_params(priority=3, bpu_cores=[0])
        self.assertEqual(
            fake.scheduling,
            {"priority": {"kws": 3}, "bpu_cores": {"kws": [0]}},
        )

    def test_from_model_gates_board_before_sdk_factory(self):
        with patch(
            "samples.speech.kws.runtime.python.kws.require_execution_target",
            side_effect=ValueError("Target mismatch"),
        ), patch(
            "utils.py_utils.single_array_runner._default_runtime_factory"
        ) as factory:
            with self.assertRaises(ValueError):
                kws_module.KWS.from_model(resolve_selection("s100"))
        factory.assert_not_called()

    def test_main_constructs_exactly_once_through_from_model(self):
        from samples.speech.kws.runtime.python.main import main

        original = kws_module.KWS.from_model
        constructed = []

        def spy_from_model(selection, *args, **kwargs):
            constructed.append(selection)
            return original(selection, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "kws.hbm"
            model.write_bytes(b"fixture model, never a real HBM")
            audio = root / "audio.wav"
            audio.write_bytes(b"fixture decoded by the injected loader")
            selection = resolve_selection(
                "s100", asset_id="s:kws:s100/kws.hbm", model_path=model
            )
            runner = kws_module.RuntimeModelRunner(selection, runtime=FakeRuntime())
            made = []

            def one_construction(sel, **kwargs):
                # A second construction would mean main (not from_model)
                # reached into the runner class again.
                assert not made, "RuntimeModelRunner constructed twice"
                made.append(sel)
                return runner

            args = [
                "--target", "s100",
                "--asset-id", selection.asset.reference,
                "--model-path", str(model),
                "--audio-file", str(root / "audio.wav"),
                "--output-dir", str(root / "output"),
            ]
            with patch(
                "utils.py_utils.platforms.require_execution_target",
                return_value="s100",
            ), patch.object(
                kws_module, "RuntimeModelRunner", side_effect=one_construction
            ), patch.object(
                kws_module.KWS, "from_model", side_effect=spy_from_model
            ), patch.object(
                audio_io, "load_audio",
                return_value=(np.ones(40000, np.float32), 16000),
            ), patch.object(
                frontend, "paddle_fbank",
                return_value=np.zeros((373, 80), np.float32),
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            self.assertEqual(len(constructed), 1)
            self.assertEqual(len(made), 1)
            report = json.loads((root / "output" / "result.json").read_text())
            self.assertTrue(report["detected"])
            self.assertTrue(report["metadata"]["model_names"])

    def test_main_reports_sdk_failure_without_result(self):
        from samples.speech.kws.runtime.python.main import main

        class FailingRuntime(FakeRuntime):
            def run(self, tensors):
                raise RuntimeError("synthetic SDK failure")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "kws.hbm"
            model.write_bytes(b"fixture model, never a real HBM")
            audio = root / "audio.wav"
            audio.write_bytes(b"fixture decoded by the injected loader")
            selection = resolve_selection(
                "s100", asset_id="s:kws:s100/kws.hbm", model_path=model
            )
            runner = kws_module.RuntimeModelRunner(
                selection, runtime=FailingRuntime()
            )
            args = [
                "--target", "s100",
                "--asset-id", selection.asset.reference,
                "--model-path", str(model),
                "--audio-file", str(root / "audio.wav"),
                "--output-dir", str(root / "output"),
            ]
            errors = io.StringIO()
            with patch(
                "utils.py_utils.platforms.require_execution_target",
                return_value="s100",
            ), patch.object(
                kws_module, "RuntimeModelRunner", return_value=runner
            ), patch.object(
                audio_io, "load_audio",
                return_value=(np.ones(40000, np.float32), 16000),
            ), patch.object(
                frontend, "paddle_fbank",
                return_value=np.zeros((373, 80), np.float32),
            ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                errors
            ):
                self.assertEqual(main(args), 2)
            self.assertIn("synthetic SDK failure", errors.getvalue())
            self.assertFalse((root / "output" / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
