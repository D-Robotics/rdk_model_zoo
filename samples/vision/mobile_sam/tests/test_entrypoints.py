# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""SDK-free entrypoint and manifest tests for MobileSAM."""

from pathlib import Path
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from samples.vision.mobile_sam.model import download
from samples.vision.mobile_sam.runtime.python import main
from samples.vision.mobile_sam.runtime.python import model_binding


ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / "samples/vision/mobile_sam"


class EntrypointTests(unittest.TestCase):
    def test_manifest_has_two_assets_for_each_published_target(self):
        for target in ("x5", "s100", "s100p", "s600"):
            with self.subTest(target=target):
                assets = model_binding.list_available_assets(target)
                self.assertEqual(len(assets), 2)
                self.assertEqual({asset.format for asset in assets}, {"bin" if target == "x5" else "hbm"})
                self.assertEqual({"encoder" in asset.filename or "decoder" in asset.filename for asset in assets}, {True})

    def test_custom_paths_require_their_exact_stage_identity(self):
        selection = model_binding.resolve_selection("s100")
        with self.assertRaises(ValueError):
            model_binding.resolve_selection("s100", decoder_model_path="/tmp/decoder.hbm")
        custom = model_binding.resolve_selection(
            "s100", decoder_model_path="/tmp/decoder.hbm",
            decoder_asset_id=selection.decoder_asset.reference,
        )
        self.assertEqual(custom.decoder_model_path, Path("/tmp/decoder.hbm"))
        with self.assertRaises(ValueError):
            model_binding.resolve_selection("s100", decoder_asset_id=selection.encoder_asset.reference)

    def test_box_parser_and_sdk_free_cli_paths(self):
        self.assertEqual(main.parse_box("185,120,380,445"), (185.0, 120.0, 380.0, 445.0))
        with self.assertRaises(SystemExit):
            main.build_parser().parse_args(["--box", "0,1,0,2"])
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

    def test_dry_run_reports_box_shape_candidates_not_observed_metadata(self):
        for target, expected in (('x5', [[1,4], [1,4,1,1]]), ('s100', [[1,4]])):
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(main.main(['--dry-run','--target',target]),0)
            decoder=json.loads(stream.getvalue())['decoder']
            self.assertEqual(decoder['boxes_shapes'],expected)
            self.assertEqual(decoder['box_shape_status'],'requires runtime metadata')

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
            "s:mobile_sam:nash-e/mobile_sam_image_encoder_norm_512x512_nashe.hbm",
            "s:mobile_sam:nash-e/mobile_sam_decoder_512_nashe.hbm",
        ])

    def test_cli_rejects_x5_core_selection_before_runner(self):
        self.assertEqual(main.main(["--target", "x5", "--bpu-cores", "0", "--dry-run"]), 2)


if __name__ == "__main__":
    unittest.main()
