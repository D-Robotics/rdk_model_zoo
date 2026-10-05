"""Entry readability: main drives task.predict through application helpers.

``main`` keeps the visible loop — warmup, one ``HimLocoTask.predict`` per
observation, action-dump recording — while ``application`` exposes the same
prepare/load/record/complete helpers its compatibility ``execute`` uses, so
there is one implementation of the evidence discipline.
"""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from samples.robotics.himloco.runtime.python import application, main
from samples.robotics.himloco.runtime.python.model_binding import (
    resolve_selection,
    SAMPLE_DIR,
)
from samples.robotics.himloco.runtime.python.policy import HimLocoTask
from samples.robotics.himloco.tests.test_binding import metadata


def fake_runtime(fail_at=None):
    m = metadata()
    runtime = MagicMock()
    runtime.model_names = m["model_names"]
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


class EntryLoopTests(unittest.TestCase):
    def prepared_args(self, root, warmup="0"):
        args = main.build_parser().parse_args(
            [
                "--target", "x5",
                "--input-path", str(SAMPLE_DIR / "test_data/obs_history"),
                "--output-dir", str(root / "out"),
                "--warmup", warmup,
            ]
        )
        model = root / "fixture.bin"
        model.write_bytes(b"synthetic model, never a real BIN")
        return args, model

    @staticmethod
    def runner_side_effect(runtime):
        from samples.robotics.himloco.runtime.python.model_runner import (
            RuntimeModelRunner,
        )

        return lambda selected: RuntimeModelRunner(
            selected, runtime_factory=lambda path: runtime
        )

    def test_main_executes_visible_predict_loop_without_execute(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, model = self.prepared_args(root, warmup="2")
            runtime, calls = fake_runtime()
            with patch(
                "samples._shared.platforms.require_execution_target"
            ), patch.object(
                application,
                "verify_asset_file",
                return_value=hashlib.sha256(model.read_bytes()).hexdigest(),
            ), patch.object(
                application,
                "RuntimeModelRunner",
                side_effect=self.runner_side_effect(runtime),
            ), patch.object(
                application, "execute", MagicMock()
            ) as execute_mock, contextlib.redirect_stdout(
                io.StringIO()
            ):
                rc = main.main(
                    [
                        "--target", "x5",
                        "--asset-id", "x5:himloco:himloco_go2_bayese_1x270.bin",
                        "--model-path", str(model),
                        "--input-path", str(SAMPLE_DIR / "test_data/obs_history"),
                        "--output-dir", str(root / "out"),
                        "--warmup", "2",
                    ]
                )
            self.assertEqual(rc, 0)
            execute_mock.assert_not_called()
            # 21 observations plus the two explicitly requested warmup runs.
            self.assertEqual(len(calls), 23)
            report = json.loads((args.output_dir / "report.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["sample_count"], 21)
            self.assertEqual(report["warmup_completed"], 2)
            self.assertEqual(len(list(args.output_dir.glob("*.bin"))), 21)


class HelperTests(unittest.TestCase):
    def prepared_run(self, root):
        args = main.build_parser().parse_args(
            [
                "--target", "x5",
                "--input-path", str(SAMPLE_DIR / "test_data/obs_history"),
                "--output-dir", str(root / "out"),
                "--warmup", "0",
            ]
        )
        model = root / "fixture.bin"
        model.write_bytes(b"synthetic model, never a real BIN")
        from dataclasses import replace

        selection = replace(
            resolve_selection("x5"), model_path=model, explicit_model_path=True
        )
        with patch("samples._shared.platforms.require_execution_target"), patch.object(
            application,
            "verify_asset_file",
            return_value=hashlib.sha256(model.read_bytes()).hexdigest(),
        ):
            run = application.prepare(args, selection)
        self.addCleanup(run.close)
        return run, selection

    def test_prepare_reserves_report_and_records_skeleton(self):
        with tempfile.TemporaryDirectory() as temp:
            run, _ = self.prepared_run(Path(temp))
            self.assertEqual(len(run.records), 21)
            self.assertIsNotNone(run.manifest)
            self.assertTrue(run.report_path.is_file())
            on_disk = json.loads(run.report_path.read_text())
            self.assertEqual(on_disk["status"], "running")
            self.assertEqual(on_disk["sample_count"], 0)
            self.assertEqual(on_disk["warmup_runs"], 0)
            self.assertIn("input_manifest", on_disk)
            # Reuse protection: a second prepare refuses the same directory.
            with self.assertRaisesRegex(ValueError, "new"):
                self.prepared_run(Path(temp))

    def test_load_task_returns_bound_task_and_records_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            run, _ = self.prepared_run(Path(temp))
            runtime, _ = fake_runtime()
            with patch.object(
                application,
                "RuntimeModelRunner",
                side_effect=EntryLoopTests.runner_side_effect(runtime),
            ):
                task = application.load_task(run)
            self.assertIsInstance(task, HimLocoTask)
            self.assertIn("runtime", run.report)
            self.assertIn("runtime_module_source", run.report)

    def test_record_sample_writes_dump_and_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            run, _ = self.prepared_run(Path(temp))
            record = run.records[0]
            digest = hashlib.sha256(record.path.read_bytes()).hexdigest()
            result = HimLocoTask(lambda feed: {
                "actions": np.arange(12, dtype=np.float32).reshape(1, 12)
            }).predict(np.zeros((1, 270), np.float32))
            destination = application.record_sample(run, record, result, digest)
            self.assertTrue(destination.is_file())
            self.assertEqual(destination.name, "000000.bin")
            entry = run.report["records"][0]
            self.assertEqual(entry["source_index"], record.source_index)
            self.assertEqual(entry["input_sha256"], digest)
            self.assertEqual(
                entry["output_sha256"], hashlib.sha256(destination.read_bytes()).hexdigest()
            )
            self.assertEqual(run.report["sample_count"], 1)

    def test_record_sample_rejects_report_path_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            run, _ = self.prepared_run(Path(temp))
            record = run.records[0]
            run.report_path = run.output_dir / "000000.bin"
            result = HimLocoTask(lambda feed: {
                "actions": np.arange(12, dtype=np.float32).reshape(1, 12)
            }).predict(np.zeros((1, 270), np.float32))
            with self.assertRaisesRegex(ValueError, "conflicts"):
                application.record_sample(run, record, result, "0" * 64)

    def test_complete_summarizes_and_marks_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            run, _ = self.prepared_run(Path(temp))
            application.complete(run, [1.0, 2.0, 3.0])
            self.assertEqual(run.report["status"], "completed")
            self.assertIsNone(run.report["current_source_index"])
            self.assertEqual(
                sorted(run.report["latency_ms"]),
                ["maximum", "mean", "minimum", "p50", "p95"],
            )
            self.assertEqual(run.report["latency_ms"]["minimum"], 1.0)
            self.assertEqual(run.report["latency_ms"]["maximum"], 3.0)

    def test_mark_failed_and_close_preserve_failure_report(self):
        with tempfile.TemporaryDirectory() as temp:
            run, _ = self.prepared_run(Path(temp))
            run.report["current_source_index"] = 4
            run.mark_failed(RuntimeError("fixture failure"))
            run.close()
            on_disk = json.loads(run.report_path.read_text())
            self.assertEqual(on_disk["status"], "failed")
            self.assertEqual(
                on_disk["error"],
                {"type": "RuntimeError", "message": "fixture failure"},
            )
            self.assertEqual(on_disk["current_source_index"], 4)
            self.assertIn("finished_utc", on_disk)
            self.assertNotIn("latency_ms", on_disk)


if __name__ == "__main__":
    unittest.main()
