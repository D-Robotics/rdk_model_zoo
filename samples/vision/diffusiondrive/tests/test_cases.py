"""Batch orchestration calls the canonical CLI and retains interrupted progress."""

import contextlib, io, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from samples.vision.diffusiondrive.runtime.python import run_cases


class CaseTests(unittest.TestCase):
    def test_dry_run_lists_five_commands_without_sdk_or_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            run_cases.single, "main"
        ) as run, contextlib.redirect_stdout(io.StringIO()) as out:
            dest = Path(tmp) / "batch"
            self.assertEqual(
                run_cases.main(
                    ["--target", "s600", "--dry-run", "--output", str(dest)]
                ),
                0,
            )
            record = json.loads(out.getvalue())
            self.assertEqual(len(record["cases"]), 5)
            self.assertFalse(dest.exists())
            run.assert_not_called()
            self.assertEqual(
                [r["case"] for r in record["cases"]], list(run_cases.CASES)
            )

    def test_batch_stops_and_records_first_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            with patch.object(
                run_cases.single, "main", side_effect=[0, 2]
            ) as run, contextlib.redirect_stdout(
                io.StringIO()
            ), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self.assertEqual(
                    run_cases.main(["--target", "s100p", "--output", str(dest)]), 2
                )
            self.assertEqual(run.call_count, 2)
            report = json.loads((dest / "batch-report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual([r["returncode"] for r in report["runs"]], [0, 2])
            self.assertEqual(report["remaining_cases"], list(run_cases.CASES[2:]))

    def test_all_cases_use_exact_target_and_separate_output_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                run_cases.single, "main", return_value=0
            ) as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    run_cases.main(
                        ["--target", "s600", "--output", str(Path(tmp) / "out")]
                    ),
                    0,
                )
            commands = [c.args[0] for c in run.call_args_list]
            self.assertEqual(len(commands), 5)
            self.assertEqual(len({c[c.index("--output") + 1] for c in commands}), 5)
            self.assertTrue(all(c[c.index("--target") + 1] == "s600" for c in commands))


if __name__ == "__main__":
    unittest.main()
