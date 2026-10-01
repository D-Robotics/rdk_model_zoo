"""Tests for the canonical ResNet source/resource integration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / "samples" / "vision" / "resnet"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResNetIntegrationTests(unittest.TestCase):
    def test_manifest_backed_downloader_resolves_and_downloads_exact_asset(self):
        from samples.vision.resnet.model import download

        observed = {}

        def fake_download(asset, destination):
            observed["asset"] = asset
            observed["destination"] = destination
            return "observed-sha"

        with mock.patch.object(download, "download_asset", fake_download):
            digest = download.download_target("s600", SAMPLE / "model")

        self.assertEqual(digest, "observed-sha")
        self.assertEqual(observed["asset"].reference,
                         "s:resnet18:s600/resnet18_224x224_nv12.hbm")
        self.assertEqual(
            observed["destination"],
            SAMPLE / "model" / "s600" / "resnet18_224x224_nv12.hbm",
        )

    def test_conversion_export_help_is_available_without_torch(self):
        script = SAMPLE / "conversion" / "export_resnet18_onnx.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--output", completed.stdout)
        self.assertIn("--opset", completed.stdout)

if __name__ == "__main__":
    unittest.main()
