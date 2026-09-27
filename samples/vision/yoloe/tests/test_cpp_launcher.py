"""Native launcher policy and run-record boundaries, without SDK or boards."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from samples.vision.yoloe.runtime.cpp import launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "float.hbm"
        self.model.write_bytes(b"host mock model")
        self.digest = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.binary = self.root / "binary"
        self.binary.write_text("fixture")
        self.binary.chmod(0o755)
        self.output = self.root / "output"
        self.args = [
            "--target",
            "s100p",
            "--variant",
            "26n",
            "--model-path",
            str(self.model),
            "--local-float-sha256",
            self.digest,
            "--binary",
            str(self.binary),
            "--output",
            str(self.output),
        ]

    def invoke(self, args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = launcher.main(args)
        return rc, out.getvalue(), err.getvalue()

    def test_dry_run_never_builds_or_claims_execution(self):
        with patch.object(
            launcher, "require_execution_target", side_effect=AssertionError
        ), patch.object(launcher.subprocess, "run", side_effect=AssertionError):
            rc, output, _ = self.invoke(self.args + ["--dry-run"])
        record = json.loads(output)
        self.assertEqual(rc, 0)
        self.assertFalse(record["executed"])
        self.assertFalse(record["runtime_metadata_verified"])
        self.assertFalse(self.output.exists())
        self.assertIn("--model-sha256", record["command"])

    def test_published_s_outputs_rejected_before_build(self):
        with patch.object(launcher.subprocess, "run") as run:
            rc, _, err = self.invoke(
                ["--target", "s100", "--build", "--output", str(self.output)]
            )
        self.assertEqual(rc, 2)
        self.assertIn("quantized", err)
        run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_identity_rejected_before_file_build_or_native_execution(self):
        with patch.object(
            launcher,
            "require_execution_target",
            side_effect=ValueError("Target mismatch"),
        ), patch.object(launcher.subprocess, "run") as run:
            rc, _, err = self.invoke(self.args)
        self.assertEqual(rc, 2)
        self.assertIn("Target mismatch", err)
        run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_model_digest_and_output_reuse_are_rejected(self):
        with patch.object(
            launcher, "require_execution_target", return_value="s100p"
        ), patch.object(launcher.subprocess, "run") as run:
            self.model.write_bytes(b"changed")
            self.assertEqual(self.invoke(self.args)[0], 2)
            self.model.write_bytes(b"host mock model")
            self.output.mkdir()
            self.assertEqual(self.invoke(self.args)[0], 2)
        run.assert_not_called()

    def mock_success(self, argv, **kwargs):
        result = Path(argv[argv.index("--output") + 1])
        result.mkdir()
        (result / "annotated.png").write_bytes(b"mock image transport")
        image = Path(argv[argv.index("--test-img") + 1])
        report = {
            "schema": "rdk-model-zoo/yoloe-native-run/v1",
            "execution_backend": "native-sdk",
            "target": "s100p",
            "variant": "26n",
            "model_sha256": self.digest,
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "vocabulary_sha256": launcher.LABELS_SHA256,
            "mask_layout": "roi",
            "image_saved": "annotated.png",
            "count": 0,
            "instances": [],
        }
        (result / "report.json").write_text(json.dumps(report))
        return subprocess.CompletedProcess(
            argv, 0, "native stdout\n", "native stderr\n"
        )

    def test_success_retains_exact_invocation_logs_and_digests(self):
        with patch.object(
            launcher, "require_execution_target", return_value="s100p"
        ), patch.object(launcher.subprocess, "run", side_effect=self.mock_success):
            self.assertEqual(self.invoke(self.args)[0], 0)
        record = json.loads((self.output / "launch-report.json").read_text())
        self.assertTrue(record["executed"])
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["model_sha256"], self.digest)
        self.assertEqual(
            record["source_asset_id"],
            launcher.resolve_selection("s100p", variant="26n").asset.reference,
        )
        self.assertFalse(record["publisher_checksum_verified"])
        self.assertEqual(
            (self.output / "native.stdout.log").read_text(), "native stdout\n"
        )
        self.assertEqual(
            (self.output / "native.stderr.log").read_text(), "native stderr\n"
        )
        self.assertIn("report.json", record["result_sha256"])
        self.assertIn("annotated.png", record["result_sha256"])

    def test_zero_exit_without_report_is_failure_with_logs(self):
        with patch.object(
            launcher, "require_execution_target", return_value="s100p"
        ), patch.object(
            launcher.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, "raw output", ""),
        ):
            self.assertEqual(self.invoke(self.args)[0], 2)
        record = json.loads((self.output / "launch-report.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["native_returncode"], 0)
        self.assertEqual((self.output / "native.stdout.log").read_text(), "raw output")

    def test_failure_preserves_native_returncode_and_stderr(self):
        with patch.object(
            launcher, "require_execution_target", return_value="s100p"
        ), patch.object(
            launcher.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 2, "", "metadata rejected\n"),
        ):
            self.assertEqual(self.invoke(self.args)[0], 2)
        record = json.loads((self.output / "launch-report.json").read_text())
        self.assertEqual(record["native_returncode"], 2)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(
            (self.output / "native.stderr.log").read_text(), "metadata rejected\n"
        )

    def test_raw_non_utf8_logs_are_preserved(self):
        with patch.object(
            launcher, "require_execution_target", return_value="s100p"
        ), patch.object(
            launcher.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 2, b"raw\xff", b"error\xfe"),
        ):
            self.assertEqual(self.invoke(self.args)[0], 2)
        self.assertEqual((self.output / "native.stdout.log").read_bytes(), b"raw\xff")
        self.assertEqual((self.output / "native.stderr.log").read_bytes(), b"error\xfe")

    def test_dangling_output_symlink_is_not_a_new_directory(self):
        self.output.symlink_to(self.root / "not-created", target_is_directory=True)
        with patch.object(
            launcher, "require_execution_target", return_value="s100p"
        ), patch.object(launcher.subprocess, "run") as run:
            self.assertEqual(self.invoke(self.args)[0], 2)
        run.assert_not_called()
        self.assertFalse((self.root / "not-created").exists())

    def test_report_identity_and_fixture_backend_cannot_claim_success(self):
        for index, (key, value) in enumerate(
            (
                ("model_sha256", "0" * 64),
                ("execution_backend", "host-fixture"),
                ("count", 1),
            )
        ):
            destination = self.root / f"bad-report-{index}"
            args = self.args.copy()
            args[args.index("--output") + 1] = str(destination)

            def wrong_report(argv, **kwargs):
                result = self.mock_success(argv, **kwargs)
                report_path = Path(argv[argv.index("--output") + 1]) / "report.json"
                report = json.loads(report_path.read_text())
                report[key] = value
                report_path.write_text(json.dumps(report))
                return result

            with patch.object(
                launcher, "require_execution_target", return_value="s100p"
            ), patch.object(launcher.subprocess, "run", side_effect=wrong_report):
                self.assertEqual(self.invoke(args)[0], 2)
            self.assertEqual(
                json.loads((destination / "launch-report.json").read_text())["status"],
                "failed",
            )

    def test_non_object_report_is_failure_with_complete_record(self):
        def wrong_report(argv, **kwargs):
            result = self.mock_success(argv, **kwargs)
            path = Path(argv[argv.index("--output") + 1]) / "report.json"
            path.write_text("[]")
            return result

        with patch.object(
            launcher, "require_execution_target", return_value="s100p"
        ), patch.object(launcher.subprocess, "run", side_effect=wrong_report):
            self.assertEqual(self.invoke(self.args)[0], 2)
        record = json.loads((self.output / "launch-report.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertIn("object", record["error"])
        self.assertEqual(record["processes"][0]["returncode"], 0)

    def test_build_failure_retains_logs_and_never_runs_native(self):
        args = self.args.copy()
        i = args.index("--binary")
        del args[i : i + 2]
        args.append("--build")
        with patch.object(
            launcher, "require_execution_target", return_value="s100p"
        ), patch.object(
            launcher.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                [], 1, b"configure output", b"SDK absent"
            ),
        ) as run:
            self.assertEqual(self.invoke(args)[0], 2)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][0], "cmake")
        record = json.loads((self.output / "launch-report.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["executed"])
        self.assertFalse(record["runtime_metadata_verified"])
        self.assertEqual(
            (self.output / "configure.stderr.log").read_bytes(), b"SDK absent"
        )
        self.assertEqual(record["processes"][0]["returncode"], 1)

    def test_e26_options_and_build_binary_conflict_rejected_on_host(self):
        self.assertEqual(
            self.invoke(self.args + ["--nms-thres", "0.7", "--dry-run"])[0], 2
        )
        self.assertEqual(self.invoke(self.args + ["--build", "--dry-run"])[0], 2)


if __name__ == "__main__":
    unittest.main()
