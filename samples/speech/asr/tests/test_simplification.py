"""Runtime-simplification boundaries for the ASR sample.

The consolidated layout keeps one readable model file: ``asr.py`` owns the
``ASR`` stage class, the raw runner construction (``RuntimeModelRunner``),
the model-owned loader ``ASR.from_model`` (runner creation, load and
binding) and the real output validation/dispatch (``transcribe``), so the
thin per-sample ``model_runner``/``postprocess`` forwarders are gone and
``main`` never assembles a runner. Host listing/dry-run must stay free of
SciPy, SoundFile and the board SDK.
"""

import contextlib
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

from samples.speech.asr.runtime.python import asr as asr_module
from samples.speech.asr.tests.test_runtime import FakeRuntime

REPO_ROOT = Path(__file__).resolve().parents[4]

LAZY_SCRIPT = """
import contextlib
import io
import sys

sys.path.insert(0, {root!r})
from samples.speech.asr.runtime.python import main

with contextlib.redirect_stdout(io.StringIO()):
    for argv in (["--list-models"], ["--target", "s100", "--dry-run"]):
        assert main.main(argv) == 0, argv
heavy = [
    name for name in ("scipy", "soundfile", "hbm_runtime")
    if name in sys.modules
]
assert not heavy, f"host modes imported heavy modules: {{heavy}}"
"""


class SimplifiedLayoutTests(unittest.TestCase):
    def test_model_file_owns_runner_and_transcribe(self):
        self.assertTrue(hasattr(asr_module, "RuntimeModelRunner"))
        self.assertTrue(hasattr(asr_module, "transcribe"))
        for removed in ("model_runner", "postprocess"):
            self.assertIsNone(
                importlib.util.find_spec(
                    f"samples.speech.asr.runtime.python.{removed}"
                ),
                f"{removed}.py should no longer exist as a forwarder module",
            )

    def test_stage_postprocess_is_real_dispatch_not_forwarding(self):
        """Integer SCALE logits decode exactly through the class entry."""
        binding = SimpleNamespace(
            output_name="logits",
            metadata=SimpleNamespace(
                output_shapes={"logits": (1, 1, 3)},
                output_dtypes={"logits": "int32"},
                output_quants={
                    "logits": SimpleNamespace(
                        quant_type="SCALE", scale=[1.0], zero_point=[0], axis=2
                    )
                },
            ),
        )
        raw = np.zeros((1, 1, 3), np.int32)
        raw[0, 0, 0] = 2**24
        raw[0, 0, 1] = 2**24 + 1
        task = asr_module.ASR(
            lambda tensors: raw, binding, ("<pad>", "a", "b")
        )
        # float32 rounding must not turn 2**24 and 2**24+1 into a tie.
        self.assertEqual(task.postprocess(raw), "a")
        self.assertEqual(task.post_process(raw), "a")

    def test_host_modes_stay_lazy(self):
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
    """ASR.from_model owns runner construction/load; main stays high-level."""

    def test_from_model_loads_and_runs_nonzero_pipeline(self):
        from samples.speech.asr.runtime.python.model_binding import (
            SAMPLE_DIR,
            resolve_selection,
        )
        from samples.speech.asr.runtime.python.vocabulary import load_vocabulary

        vocabulary = load_vocabulary(SAMPLE_DIR / "test_data/vocab.json")
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model.hbm"
            model.write_bytes(b"fixture model, never a real HBM")
            selection = resolve_selection(
                "s100", asset_id="s:asr:s100/asr.hbm", model_path=model
            )
            fake = FakeRuntime()
            task = asr_module.ASR.from_model(selection, vocabulary, runtime=fake)
            self.assertIsInstance(task, asr_module.ASR)
            # Nonzero stereo 8 kHz input: mono mix, Fourier resample, z-score
            # and padding all run before the single raw model call.
            stereo = np.stack(
                [np.linspace(-1, 1, 800, dtype=np.float32), np.zeros(800, np.float32)],
                axis=1,
            )
            self.assertEqual(task.predict(stereo, 8000), vocabulary[5] * 2)
            self.assertEqual(fake.calls, 1)
            self.assertEqual(task.metadata.output_names, ("logits",))
            task.set_scheduling_params(priority=2, bpu_cores=[0])
            self.assertEqual(
                fake.scheduling,
                {"priority": {"asr": 2}, "bpu_cores": {"asr": [0]}},
            )

    def test_from_model_gates_board_before_sdk_factory(self):
        from samples.speech.asr.runtime.python.model_binding import resolve_selection

        with patch(
            "samples.speech.asr.runtime.python.asr.require_execution_target",
            side_effect=ValueError("Target mismatch"),
        ), patch(
            "utils.py_utils.single_array_runner._default_runtime_factory"
        ) as factory:
            with self.assertRaises(ValueError):
                asr_module.ASR.from_model(
                    resolve_selection("s100"), ("<pad>", "a")
                )
        factory.assert_not_called()

    def test_main_constructs_exactly_once_through_from_model(self):
        from samples.speech.asr.runtime.python import audio_io
        from samples.speech.asr.runtime.python.main import main
        from samples.speech.asr.runtime.python.model_binding import resolve_selection

        original = asr_module.ASR.from_model
        constructed = []

        def spy_from_model(selection, *args, **kwargs):
            constructed.append(selection)
            return original(selection, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.hbm"
            model.write_bytes(b"fixture model, never a real HBM")
            selection = resolve_selection(
                "s100", asset_id="s:asr:s100/asr.hbm", model_path=model
            )
            runner = asr_module.RuntimeModelRunner(selection, runtime=FakeRuntime())

            def chunks(*args):
                yield audio_io.AudioChunk(
                    np.array([0.1, 0.2], np.float32), 16000, 0, 0
                )
                yield audio_io.AudioChunk(
                    np.array([0.3], np.float32), 16000, 1, 1
                )

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
                "--output-dir", str(root / "out"),
            ]
            with patch(
                "utils.py_utils.platforms.require_execution_target",
                return_value="s100",
            ), patch.object(
                asr_module, "RuntimeModelRunner", side_effect=one_construction
            ), patch.object(
                asr_module.ASR, "from_model", side_effect=spy_from_model
            ), patch.object(
                audio_io, "read_chunks", side_effect=chunks
            ), contextlib.redirect_stdout(
                io.StringIO()
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 0)
            self.assertEqual(len(constructed), 1)
            self.assertEqual(len(made), 1)
            report = json.loads((root / "out" / "result.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["chunks"]), 2)
            self.assertTrue(report["metadata"]["model_names"])


if __name__ == "__main__":
    unittest.main()
