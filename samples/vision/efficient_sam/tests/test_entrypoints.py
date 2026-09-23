# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""SDK-free entrypoint and manifest tests for EfficientSAM."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from samples.vision.efficient_sam.model import download
from samples.vision.efficient_sam.runtime.python import main
from samples.vision.efficient_sam.runtime.python import model_binding


ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / "samples/vision/efficient_sam"


class EntrypointTests(unittest.TestCase):
    def test_manifest_has_two_assets_for_each_published_target(self):
        for target in ("x5", "s100", "s100p", "s600"):
            with self.subTest(target=target):
                assets = model_binding.list_available_assets(target)
                self.assertEqual(len(assets), 2)
                self.assertEqual({asset.format for asset in assets}, {"bin" if target == "x5" else "hbm"})
                self.assertEqual({asset.filename.split("/")[-1].split("_")[0] for asset in assets}, {"efficient"})

    def test_custom_paths_require_their_exact_stage_identity(self):
        selection = model_binding.resolve_selection("s100")
        with self.assertRaises(ValueError):
            model_binding.resolve_selection("s100", encoder_model_path="/tmp/encoder.hbm")
        custom = model_binding.resolve_selection(
            "s100", encoder_model_path="/tmp/encoder.hbm",
            encoder_asset_id=selection.encoder_asset.reference,
        )
        self.assertEqual(custom.encoder_model_path, Path("/tmp/encoder.hbm"))
        with self.assertRaises(ValueError):
            model_binding.resolve_selection("s100", encoder_asset_id=selection.decoder_asset.reference)

    def test_sdk_free_cli_paths(self):
        for argv in (("--help",), ("--list-models",), ("--dry-run", "--target", "x5")):
            with self.subTest(argv=argv):
                result = subprocess.run(
                    [sys.executable, str(SAMPLE / "runtime/python/main.py"), *argv],
                    cwd="/tmp", capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [sys.executable, str(SAMPLE / "runtime/python/main.py"), "--dry-run"],
            cwd="/tmp", capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)

    def test_download_delegates_manifest_pair_without_network(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory, patch.object(
            download, "download_asset",
            side_effect=lambda asset, path: calls.append((asset.reference, Path(path))) or "fixture-digest",
        ), redirect_stdout(StringIO()) as output:
            result = download.download_target("s100", directory)
        self.assertEqual(len(result), 2)
        self.assertEqual(result, ("fixture-digest", "fixture-digest"))
        self.assertEqual(output.getvalue().count("observed_sha256=fixture-digest"), 2)
        self.assertEqual([reference for reference, _ in calls], [
            "s:efficient_sam:nash-e/efficient_sam_vitt_encoder_512x512_nashe.hbm",
            "s:efficient_sam:nash-e/efficient_sam_vitt_decoder_512_nashe.hbm",
        ])

    def test_cli_rejects_x5_core_selection_before_runner(self):
        self.assertEqual(main.main(["--target", "x5", "--bpu-cores", "0", "--dry-run"]), 2)


if __name__ == "__main__":
    unittest.main()
