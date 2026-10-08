"""Offline CLI integration with explicitly synthetic SDK/model bytes."""

import contextlib
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from samples.robotics.himloco.runtime.python import cli, input_io, main, policy
from samples.robotics.himloco.runtime.python.model_binding import (
    resolve_selection,
    SAMPLE_DIR,
)
from samples.robotics.himloco.runtime.python.policy import RuntimeModelRunner
from samples.robotics.himloco.tests.test_binding import metadata


class CliTests(unittest.TestCase):
    def argv(self, root):
        return [
            "--target",
            "x5",
            "--asset-id",
            "x5:himloco:himloco_go2_bayese_1x270.bin",
            "--model-path",
            str(root / "fixture.bin"),
            "--input-path",
            str(SAMPLE_DIR / "test_data/obs_history"),
            "--output-dir",
            str(root / "out"),
            "--warmup",
            "2",
        ]

    def selection(self, root):
        model = root / "fixture.bin"
        model.write_bytes(b"synthetic model, never a real BIN")
        return replace(
            resolve_selection("x5"), model_path=model, explicit_model_path=True
        )

    def runtime(self, fail_at=None):
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
        calls = []

        def run(feed):
            calls.append(feed)
            if fail_at is not None and len(calls) == fail_at:
                raise RuntimeError("synthetic SDK failure")
            return {
                "policy": {"actions": np.arange(12, dtype=np.float32).reshape(1, 12)}
            }

        runtime.run = run
        runtime.set_scheduling_params = Mock()
        return runtime, calls

    def run_fake(self, argv, root, runtime):
        with patch("utils.py_utils.platforms.require_execution_target"), patch.object(
            cli,
            "verify_asset_file",
            return_value=hashlib.sha256(
                (root / "fixture.bin").read_bytes()
            ).hexdigest(),
        ), patch.object(
            policy,
            "RuntimeModelRunner",
            side_effect=lambda selected, **kwargs: RuntimeModelRunner(
                selected, runtime_factory=lambda path: runtime
            ),
        ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            rc = main.main(argv)
        report = json.loads((root / "out" / "report.json").read_text())
        return rc, report

    def test_all_source_inputs_warmup_dumps_and_reuse_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.selection(root)
            runtime, calls = self.runtime()
            rc, report = self.run_fake(self.argv(root), root, runtime)
            self.assertEqual(rc, 0)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["sample_count"], 21)
            self.assertEqual(len(calls), 23)
            self.assertEqual(len(list((root / "out").glob("*.bin"))), 21)
            np.testing.assert_array_equal(
                np.fromfile(root / "out" / "000000.bin", dtype="<f4"),
                np.arange(12, dtype=np.float32),
            )
            self.assertEqual(
                report["input_manifest"]["source_sha256"],
                "49f5459a5ff4d8003d9ee9d95c1104d158688017408a74bcd1506ff171cc01ab",
            )
            before = (root / "out" / "report.json").read_bytes()
            rc, _ = self.run_fake(self.argv(root), root, runtime)
            self.assertEqual(rc, 2)
            self.assertEqual(before, (root / "out" / "report.json").read_bytes())

    def test_failure_retains_completed_outputs_and_current_source_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.selection(root)
            runtime, _ = self.runtime(fail_at=4)
            rc, report = self.run_fake(self.argv(root), root, runtime)
            self.assertEqual(rc, 2)
            self.assertEqual(report["status"], "failed")
            self.assertIn("synthetic SDK failure", report["error"]["message"])
            self.assertEqual(report["sample_count"], 1)
            self.assertEqual(report["current_source_index"], 1)
            self.assertNotIn("latency_ms", report)

    def test_host_preview_does_not_load_sdk_or_write(self):
        for flags in (["--list-models"], ["--target", "x5", "--dry-run"]):
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(main.main(flags), 0)
            self.assertFalse(json.loads(stream.getvalue())["sdk_loaded"])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main.main(["--target", "s100", "--dry-run"]), 2)

    def test_duplicate_indices_and_bad_manifest_digest_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "obs_history"
            directory.mkdir()
            data = np.zeros(270, dtype="<f4").tobytes()
            (directory / "0.bin").write_bytes(data)
            (directory / "000000.bin").write_bytes(data)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                input_io.discover_inputs(directory)
            (directory / "0.bin").unlink()
            manifest = json.loads(
                (SAMPLE_DIR / "test_data/runtime-input-manifest.json").read_text()
            )
            manifest["records"] = manifest["records"][:1]
            (root / "runtime-input-manifest.json").write_text(json.dumps(manifest))
            records, _ = input_io.discover_inputs(directory)
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                input_io.load_observation(records[0])


if __name__ == "__main__":
    unittest.main()
