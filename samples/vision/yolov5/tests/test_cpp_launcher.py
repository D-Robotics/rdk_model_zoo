# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Host-only launcher checks; native SDK execution is intentionally absent."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = ROOT / "samples/vision/yolov5/runtime/cpp/launcher.py"


class CppLauncherTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, str(LAUNCHER), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

    def test_help_and_list_are_sdk_free(self):
        help_result = self.invoke("--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--asset-id", help_result.stdout)
        listed = self.invoke("--list-models", "--target", "s100")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(listed.stdout.strip(), "s:yolov5:s100/yolov5x_672x672_nv12.hbm")
        x5 = self.invoke("--list-models", "--target", "x5")
        self.assertEqual(x5.returncode, 0, x5.stderr)
        self.assertEqual(len(x5.stdout.splitlines()), 9)

    def test_dry_run_is_explicit_and_reports_exact_publication(self):
        result = self.invoke("--dry-run", "--target", "s600")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["target"], "s600")
        self.assertEqual(payload["asset_id"], "s:yolov5:s600/yolov5x_672x672_nv12.hbm")
        self.assertEqual(payload["filename"], "s600/yolov5x_672x672_nv12.hbm")
        self.assertEqual(payload["board_status"], "not-run")

    def test_external_path_requires_exact_asset_id_and_s100p_has_no_asset(self):
        missing_id = self.invoke("--dry-run", "--target", "x5", "--model-path", "/tmp/model.bin")
        self.assertEqual(missing_id.returncode, 2)
        self.assertIn("requires an exact asset-id", missing_id.stderr)
        bad_id = self.invoke("--dry-run", "--target", "x5", "--asset-id", "s:yolov5:s100/x.hbm")
        self.assertEqual(bad_id.returncode, 2)
        s100p = self.invoke("--dry-run", "--target", "s100p")
        self.assertEqual(s100p.returncode, 2)
        self.assertIn("No published", s100p.stderr)

    def test_dry_run_rejects_invalid_runtime_parameters(self):
        threshold = self.invoke("--dry-run", "--target", "x5", "--score-thres", "1.1")
        self.assertEqual(threshold.returncode, 2)
        core = self.invoke("--dry-run", "--target", "x5", "--bpu-core", "-2")
        self.assertEqual(core.returncode, 2)

    def test_x5_defaults_to_the_cpp_source_variant_not_the_python_default(self):
        # The fixed X5 C++ source (runtime/cpp/main.cc) defaults to s-v2.0 while
        # the unified Python runtime defaults to n-v7.0; the native launcher must
        # keep the C++ source default rather than silently inheriting the other.
        default = self.invoke("--dry-run", "--target", "x5")
        self.assertEqual(default.returncode, 0, default.stderr)
        payload = json.loads(default.stdout)
        self.assertEqual(payload["variant"], "s-v2.0")
        self.assertEqual(
            payload["filename"], "yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin"
        )
        python_default = self.invoke("--dry-run", "--target", "x5", "--variant", "n-v7.0")
        self.assertEqual(python_default.returncode, 0, python_default.stderr)
        self.assertEqual(
            json.loads(python_default.stdout)["variant"], "n-v7.0"
        )

    def test_x5_rejects_unapplied_scheduling_parameters(self):
        priority = self.invoke("--dry-run", "--target", "x5", "--priority", "3")
        self.assertEqual(priority.returncode, 2)
        self.assertIn("no verified HB-DNN scheduling mapping", priority.stderr)
        core = self.invoke("--dry-run", "--target", "x5", "--bpu-core", "2")
        self.assertEqual(core.returncode, 2)
        # The S adapter does apply them, so the same values stay valid there.
        s_target = self.invoke("--dry-run", "--target", "s600", "--priority", "3", "--bpu-core", "2")
        self.assertEqual(s_target.returncode, 0, s_target.stderr)


if __name__ == "__main__":
    unittest.main()
