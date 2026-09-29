"""Real CLI workflows with an injected SDK; no model downloads or board calls."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from samples.vision.yolo26_depth.runtime.python import main
from samples.vision.yolo26_depth.model import download
from samples.vision.yolo26_depth.runtime.python.model_binding import resolve_selection
from samples.vision.yolo26_depth.runtime.python.model_runner import RuntimeModelRunner
from test_depth import metadata, ROOT


class CliTests(unittest.TestCase):
    def test_sdk_free_list_dry_run_and_validation(self):
        path = ROOT / "samples/vision/yolo26_depth/runtime/python/main.py"
        for argv, count in (
            (["--list-models"], 20),
            (["--target", "s100p", "--variant", "l", "--dry-run"], None),
        ):
            run = subprocess.run(
                [sys.executable, str(path), *argv],
                capture_output=True,
                text=True,
                cwd="/tmp",
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            value = json.loads(run.stdout)
            if count:
                self.assertEqual(len(value), count)
            else:
                self.assertEqual(value["profile"], "lite")
                self.assertFalse(value["sdk_loaded"])
        for argv in (["--warmup", "-1"], ["--priority", "256"], ["--bpu-cores", "-1"]):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main.main(["--target", "x5", "--dry-run", *argv]), 2)

    def test_cli_full_workflow_preserves_raw_outputs_and_warmup_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            image = p / "image.png"
            cv2.imwrite(str(image), np.zeros((17, 29, 3), np.uint8))
            model = p / "fixture.hbm"
            model.write_bytes(b"host fake artifact")
            s = resolve_selection("s100p", variant="l")
            raw = np.linspace(-5, 6, 192 * 192, dtype=np.float32).reshape(
                1, 192, 192, 1
            )
            runs = []
            schedules = []
            m = metadata(True)
            runtime = SimpleNamespace(
                **{k: v for k, v in m.items() if k != "model_name"},
                version="host-fixture",
                run=lambda inputs: (runs.append(inputs) or {"depth": {"output0": raw}}),
                set_scheduling_params=lambda **kw: schedules.append(kw)
            )
            real = RuntimeModelRunner
            with patch(
                "samples.vision.yolo26_depth.runtime.python.model_runner.RuntimeModelRunner",
                side_effect=lambda selection: real(selection, runtime=runtime),
            ), contextlib.redirect_stdout(io.StringIO()):
                rc = main.main(
                    [
                        "--asset-id",
                        s.asset.reference,
                        "--model-path",
                        str(model),
                        "--test-img",
                        str(image),
                        "--output",
                        str(p / "out"),
                        "--warmup",
                        "2",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(len(runs), 3)
            self.assertEqual(
                schedules, [{"priority": {"depth": 0}, "bpu_cores": {"depth": [0]}}]
            )
            report = json.loads((p / "out/report.json").read_text())
            self.assertEqual(report["runtime_version"], "host-fixture")
            self.assertEqual(report["target"], "s100p")
            self.assertEqual(report["profile"], "lite")
            self.assertEqual(report["depth_native_shape"], [17, 29])
            self.assertEqual(report["warmup"], 2)
            self.assertEqual(len(report["model_sha256"]), 64)
            np.testing.assert_array_equal(
                np.load(p / "out/raw_logit.npy"), raw.squeeze()
            )
            self.assertIsNotNone(cv2.imread(str(p / "out/overlay.png")))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main.main(["--target", "x5", "--output", str(p / "out")]), 2
                )

    def test_gate_precedes_factory(self):
        with patch(
            "samples.vision.yolo26_depth.runtime.python.model_runner.require_execution_target",
            side_effect=ValueError("wrong board"),
        ) as gate:
            with self.assertRaisesRegex(ValueError, "wrong board"):
                RuntimeModelRunner(resolve_selection("s100p")).load()
            gate.assert_called_once_with("s100p")

    def test_converted_mode_keeps_target_gate_and_published_hash_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "converted.bin"
            p.write_bytes(b"custom conversion fixture")
            original = resolve_selection("x5")
            published = resolve_selection(
                "x5", asset_id=original.asset.reference, model_path=p
            )
            custom = resolve_selection(
                "x5",
                asset_id=original.asset.reference,
                model_path=p,
                converted_model=True,
            )
            m = metadata(False)
            runtime = SimpleNamespace(
                **{k: v for k, v in m.items() if k != "model_name"}
            )
            calls = []
            sdk = SimpleNamespace(
                HB_HBMRuntime=lambda path: (calls.append(path) or runtime)
            )
            with patch.dict(sys.modules, {"hbm_runtime": sdk}), patch(
                "samples.vision.yolo26_depth.runtime.python.model_runner.require_execution_target"
            ) as gate:
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    RuntimeModelRunner(published).load()
                self.assertEqual(calls, [])
                RuntimeModelRunner(custom).load()
                self.assertEqual(calls, [str(p)])
                self.assertEqual(gate.call_count, 2)
            with patch(
                "samples.vision.yolo26_depth.runtime.python.model_runner.require_execution_target",
                side_effect=ValueError("wrong board"),
            ):
                with self.assertRaisesRegex(ValueError, "wrong board"):
                    RuntimeModelRunner(custom).load()
            with self.assertRaises(ValueError):
                resolve_selection("x5", converted_model=True)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    main.main(
                        [
                            "--asset-id",
                            original.asset.reference,
                            "--model-path",
                            str(p),
                            "--converted-model",
                            "--dry-run",
                        ]
                    ),
                    0,
                )
            report = json.loads(output.getvalue())
            self.assertIsNone(report["asset_id"])
            self.assertEqual(report["contract_reference"], original.asset.reference)
            self.assertEqual(report["artifact_origin"], "user-converted")

    def test_download_exact_asset_preserves_hash_policy(self):
        for target in ("x5", "s100p"):
            with patch.object(
                download, "download_asset", return_value="a" * 64
            ) as call, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    download.main(
                        [
                            "--target",
                            target,
                            "--variant",
                            "x",
                            "--output-dir",
                            "/tmp/depth-fixture",
                        ]
                    ),
                    0,
                )
            asset, path = call.call_args.args
            self.assertEqual(asset, resolve_selection(target, variant="x").asset)
            self.assertEqual(path, Path("/tmp/depth-fixture") / asset.filename)
            self.assertEqual(asset.sha256 is None, target != "x5")


if __name__ == "__main__":
    unittest.main()
