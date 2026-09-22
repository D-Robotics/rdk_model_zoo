"""Download-target resolution tests for the EfficientFormerV2 sample.

The variant must be forwarded into the manifest reference.  These tests pin
the exact published reference every supported (target, variant) combination
must resolve to — both rows — with the network downloader stubbed out.
"""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

from samples.vision.efficientformerv2.model import download as download_mod

EXPECTED_REFERENCES = {
    ('x5', 's0'): 'x5:efficientformerv2:EfficientFormerv2_s0_224x224_nv12.bin',
    ('x5', 's1'): 'x5:efficientformerv2:EfficientFormerv2_s1_224x224_nv12.bin',
    ('x5', 's2'): 'x5:efficientformerv2:EfficientFormerv2_s2_224x224_nv12.bin',
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

    def test_default_download_target_is_l3_on_x5(self):
        captured = {}

        def fake_download(asset, destination):
            captured["reference"] = asset.reference
            return "1" * 64

        with mock.patch.object(download_mod, "download_asset", fake_download):
            download_mod.download_target("x5", self.output_dir)
        # The source entrypoint defaulted to the S0 artifact.
        self.assertEqual(
            captured["reference"], "x5:efficientformerv2:EfficientFormerv2_s0_224x224_nv12.bin"
        )

    def test_unpublished_combination_is_an_explicit_error(self):
        with self.assertRaises(ValueError):
            download_mod.download_target("s100", self.output_dir, variant="s1")


if __name__ == "__main__":
    unittest.main()
