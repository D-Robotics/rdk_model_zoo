# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Small executable checks for the FCOS public documentation contract."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / "samples" / "vision" / "fcos"


class ReadmeContractTests(unittest.TestCase):
    def test_all_required_bilingual_anchors_and_local_links_exist(self):
        required = {
            "README.md": ("overview", "support-matrix", "prerequisites", "quickstart", "expected-results", "directory", "entry-points", "license"),
            "model/README.md": ("artifacts", "preparation", "accompanying-files", "local-paths", "formats-checksums"),
            "runtime/python/README.md": ("environment", "usage", "parameters", "results", "integration-example", "stage-io", "troubleshooting"),
            "conversion/README.md": ("source-model", "toolchain-targets", "export", "calibration", "compile", "validation", "artifacts", "known-gaps"),
            "evaluator/README.md": ("dataset", "environment", "command", "metrics", "outputs", "reference-results", "boundaries"),
        }
        for relative, anchors in required.items():
            for language in (relative, relative.replace("README.md", "README_cn.md")):
                path = SAMPLE / language
                text = path.read_text(encoding="utf-8")
                for anchor in anchors:
                    self.assertIn(f'<a id="{anchor}"></a>', text, f"missing {anchor} in {path}")
                for target in re.findall(r"\]\(([^)#]+)(?:#[^)]+)?\)", text):
                    if target.startswith(("http:", "https:")):
                        continue
                    self.assertTrue((path.parent / target).resolve().exists(), f"broken link {target} in {path}")

    def test_runtime_readme_parameters_match_parser_defaults(self):
        from samples.vision.fcos.runtime.python.main import build_parser

        args = build_parser().parse_args([])
        self.assertIsNone(args.variant)
        self.assertEqual(args.classes_num, 80)
        self.assertEqual(args.conf_thres, 0.5)
        self.assertEqual(args.iou_thres, 0.6)
        docs = (SAMPLE / "runtime/python/README.md").read_text(encoding="utf-8")
        self.assertIn("| `--variant` | str | `None`", docs)
        self.assertIn("| `--conf-thres` | float | `0.5`", docs)
        self.assertIn("| `--iou-thres` | float | `0.6`", docs)


if __name__ == "__main__":
    unittest.main()
