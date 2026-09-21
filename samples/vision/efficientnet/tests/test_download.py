"""Download-target resolution tests for the EfficientNet sample.

Pattern carried over from the board-found MobileNetV4 defect (B1-D2): the
variant must be forwarded into the manifest reference.  These tests pin the
exact published reference every supported (target, variant) combination must
resolve to — all 13 rows — with the network downloader stubbed out.
"""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

from samples.vision.efficientnet.model import download as download_mod

EXPECTED_REFERENCES = {
    ('x5', 'b2'): 'x5:efficientnet:EfficientNet_B2_224x224_nv12.bin',
    ('x5', 'b3'): 'x5:efficientnet:EfficientNet_B3_224x224_nv12.bin',
    ('x5', 'b4'): 'x5:efficientnet:EfficientNet_B4_224x224_nv12.bin',
    ('s100', 'lite0'): 's:efficientnet:s100/efficientnet_lite0_224x224_nv12.hbm',
    ('s100', 'lite1'): 's:efficientnet:s100/efficientnet_lite1_240x240_nv12.hbm',
    ('s100', 'lite2'): 's:efficientnet:s100/efficientnet_lite2_260x260_nv12.hbm',
    ('s100', 'lite3'): 's:efficientnet:s100/efficientnet_lite3_300x300_nv12.hbm',
    ('s100', 'lite4'): 's:efficientnet:s100/efficientnet_lite4_380x380_nv12.hbm',
    ('s600', 'lite0'): 's:efficientnet:s600/efficientnet_lite0_224x224_nv12.hbm',
    ('s600', 'lite1'): 's:efficientnet:s600/efficientnet_lite1_240x240_nv12.hbm',
    ('s600', 'lite2'): 's:efficientnet:s600/efficientnet_lite2_260x260_nv12.hbm',
    ('s600', 'lite3'): 's:efficientnet:s600/efficientnet_lite3_300x300_nv12.hbm',
    ('s600', 'lite4'): 's:efficientnet:s600/efficientnet_lite4_380x380_nv12.hbm',
}


class DownloadTargetReferenceTests(unittest.TestCase):
    output_dir = Path("/tmp/rdk-zoo-download-fixture")

    def test_download_target_resolves_the_published_reference_per_target_variant(self):
        for (target, variant), expected in EXPECTED_REFERENCES.items():
            with self.subTest(target=target, variant=variant):
                captured = {}

                def fake_download(asset, destination):
                    captured["reference"] = asset.reference
                    captured["filename"] = asset.filename
                    captured["destination"] = destination
                    return "0" * 64

                with mock.patch.object(download_mod, "download_asset", fake_download):
                    digest = download_mod.download_target(
                        target, self.output_dir, variant=variant
                    )
                self.assertEqual(captured["reference"], expected)
                self.assertEqual(captured["filename"], expected.rsplit(":", 1)[1])
                self.assertEqual(
                    captured["destination"], self.output_dir / captured["filename"]
                )
                self.assertEqual(digest, "0" * 64)

    def test_default_download_target_follows_the_target(self):
        """B2-R1 regression: an omitted variant resolves each target's source
        default (X5 b2, S lite0) instead of a global b2 that fails on S."""

        expected = {
            "x5": "x5:efficientnet:EfficientNet_B2_224x224_nv12.bin",
            "s100": "s:efficientnet:s100/efficientnet_lite0_224x224_nv12.hbm",
            "s600": "s:efficientnet:s600/efficientnet_lite0_224x224_nv12.hbm",
        }
        for target, reference in expected.items():
            with self.subTest(target=target):
                captured = {}

                def fake_download(asset, destination):
                    captured["reference"] = asset.reference
                    return "0" * 64

                with mock.patch.object(download_mod, "download_asset", fake_download):
                    download_mod.download_target(target, self.output_dir)
                self.assertEqual(captured["reference"], reference)

    def test_default_variants_match_the_binding_table(self):
        """The downloader's per-target defaults must not drift from the binding
        table's declared defaults (single source of truth for the contract)."""

        from samples.vision.efficientnet.runtime.python.model_binding import (
            BINDING_TABLE,
        )

        for target, variant in download_mod.DEFAULT_VARIANTS.items():
            self.assertEqual(BINDING_TABLE.default_variant_for(target), variant)

    def test_unpublished_combination_is_rejected_without_network(self):
        with self.assertRaises(ValueError) as ctx:
            download_mod.asset_reference("x5", "lite0")
        self.assertIn("Unsupported EfficientNet target/variant", str(ctx.exception))

    def test_target_without_default_is_rejected_explicitly(self):
        with self.assertRaises(ValueError) as ctx:
            download_mod.asset_reference("s100p")
        self.assertIn("no default variant", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
