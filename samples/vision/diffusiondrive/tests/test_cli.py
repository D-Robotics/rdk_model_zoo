"""Actual planning CLI through injected named runtime; no model/SDK download."""

import contextlib, io, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from samples.vision.diffusiondrive.runtime.python import main
from samples.vision.diffusiondrive.runtime.python.model_binding import resolve_selection
from samples.vision.diffusiondrive.runtime.python.model_runner import RuntimeModelRunner
from test_diffusiondrive import metadata, arrays, SOURCE


class CliTests(unittest.TestCase):
    def test_inspection_and_invalid_target_do_not_load_runtime(self):
        with patch.object(main, "RuntimeModelRunner") as runner:
            for args in (["--list-models"], ["--target", "s600", "--dry-run"]):
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(main.main(args), 0)
                    self.assertTrue(json.loads(out.getvalue()))
            for args in (
                ["--target", "s100"],
                ["--target", "s100p", "--agent-score-thres", "nan"],
                ["--target", "s600", "--priority", "256"],
            ):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main.main(args + ["--dry-run"]), 2)
            runner.assert_not_called()

    def test_full_cli_archives_inputs_raw_outputs_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            model = p / "fixture.hbm"
            model.write_bytes(b"fixture-only")
            m = metadata()
            raw = arrays("reference_outputs.npz")
            schedules = []
            runtime = SimpleNamespace(
                **{k: v for k, v in m.items() if k != "model_name"},
                run=lambda data: {"plan": raw},
                set_scheduling_params=lambda **kw: schedules.append(kw)
            )
            with patch.object(
                main,
                "RuntimeModelRunner",
                side_effect=lambda s: RuntimeModelRunner(s, runtime=runtime),
            ), contextlib.redirect_stdout(io.StringIO()):
                rc = main.main(
                    [
                        "--platform",
                        "s600",
                        "--asset-id",
                        resolve_selection("s600").asset.reference,
                        "--model-path",
                        str(model),
                        "--input-npz",
                        str(SOURCE / "test_data/reference_inputs.npz"),
                        "--output",
                        str(p / "out"),
                        "--output-npz",
                        str(p / "extra.npz"),
                        "--output-image",
                        str(p / "extra.jpg"),
                    ]
                )
            self.assertEqual(rc, 0)
            report = json.loads((p / "out/report.json").read_text())
            self.assertEqual(set(report["runtime_metadata"]["input_quants"]), set())
            self.assertEqual(
                report["input_sha256"],
                main.sha256_file(SOURCE / "test_data/reference_inputs.npz"),
            )
            self.assertEqual(report["scheduling"], {"priority": 0, "bpu_cores": [0]})
            self.assertEqual(
                schedules, [{"priority": {"plan": 0}, "bpu_cores": {"plan": [0]}}]
            )
            with np.load(p / "out/raw_outputs.npz", allow_pickle=False) as data:
                for name in raw:
                    np.testing.assert_array_equal(data[name], raw[name])
            with np.load(p / "out/physical_inputs.npz", allow_pickle=False) as data:
                for name, value in arrays("reference_inputs.npz").items():
                    np.testing.assert_array_equal(data[name], value)
            self.assertEqual(
                (p / "out/outputs.npz").read_bytes(), (p / "extra.npz").read_bytes()
            )
            self.assertTrue((p / "extra.jpg").read_bytes().startswith(b"\xff\xd8"))

    def test_collision_and_bad_features_fail_before_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            features = arrays("reference_inputs.npz")
            features["status"] = np.zeros((8,), np.float32)
            np.savez(p / "bad.npz", **features)
            with patch.object(
                main, "RuntimeModelRunner"
            ) as runner, contextlib.redirect_stderr(io.StringIO()):
                for args in (
                    ["--input-npz", str(p / "bad.npz")],
                    ["--output-image", str(p / "extra.txt")],
                    ["--output-npz", str(p / "out/raw_outputs.npz")],
                    ["--output-image", str(p / "out")],
                    [
                        "--output-image",
                        str(p / "same"),
                        "--output-npz",
                        str(p / "same"),
                    ],
                ):
                    self.assertEqual(
                        main.main(
                            ["--target", "s600", "--output", str(p / "out"), *args]
                        ),
                        2,
                    )
                runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
