"""Download-target resolution tests for the ResNet sample.

Regression coverage for the board-found defect (X5 smoke, 2026-09-21) where
``download_target`` resolved its manifest reference without forwarding the
variant, so a non-default ``--variant`` fetched the default artifact.  These tests pin
the exact published reference every supported combination must resolve to,
with the network downloader stubbed out.
"""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

from samples.vision.resnet.model import download as download_mod

EXPECTED_REFERENCES = {
    ('x5', '18'): 'x5:resnet:resnet18_bayese_224x224_nv12.bin',
    ('s100', '18'): 's:resnet:s100/resnet18_nashe_224x224_nv12.hbm',
    ('s100p', '18'): 's:resnet:s100p/resnet18_nashm_224x224_nv12.hbm',
    ('s600', '18'): 's:resnet:s600/resnet18_nashp_224x224_nv12.hbm',
    ('x5', '34'): 'x5:resnet:resnet34_bayese_224x224_nv12.bin',
    ('s100', '34'): 's:resnet:s100/resnet34_nashe_224x224_nv12.hbm',
    ('s100p', '34'): 's:resnet:s100p/resnet34_nashm_224x224_nv12.hbm',
    ('s600', '34'): 's:resnet:s600/resnet34_nashp_224x224_nv12.hbm',
    ('x5', '50'): 'x5:resnet:resnet50_bayese_224x224_nv12.bin',
    ('s100', '50'): 's:resnet:s100/resnet50_nashe_224x224_nv12.hbm',
    ('s100p', '50'): 's:resnet:s100p/resnet50_nashm_224x224_nv12.hbm',
    ('s600', '50'): 's:resnet:s600/resnet50_nashp_224x224_nv12.hbm',
    ('x5', '101'): 'x5:resnet:resnet101_bayese_224x224_nv12.bin',
    ('s100', '101'): 's:resnet:s100/resnet101_nashe_224x224_nv12.hbm',
    ('s100p', '101'): 's:resnet:s100p/resnet101_nashm_224x224_nv12.hbm',
    ('s600', '101'): 's:resnet:s600/resnet101_nashp_224x224_nv12.hbm',
    ('x5', '152'): 'x5:resnet:resnet152_bayese_224x224_nv12.bin',
    ('s100', '152'): 's:resnet:s100/resnet152_nashe_224x224_nv12.hbm',
    ('s100p', '152'): 's:resnet:s100p/resnet152_nashm_224x224_nv12.hbm',
    ('s600', '152'): 's:resnet:s600/resnet152_nashp_224x224_nv12.hbm',
}


class DownloadTargetReferenceTests(unittest.TestCase):
    output_dir = Path("/tmp/rdk-zoo-download-fixture")

    def test_download_target_resolves_the_published_reference_per_target_and_variant(self):
        for target, variant in EXPECTED_REFERENCES:
            expected = EXPECTED_REFERENCES[(target, variant)]
            with self.subTest(target=target, variant=variant):
                captured = {}

                def fake_download(asset, destination):
                    captured["reference"] = asset.reference
                    captured["filename"] = asset.filename
                    captured["destination"] = destination
                    return "0" * 64

                with mock.patch.object(download_mod, "download_asset", fake_download):
                    digest = download_mod.download_target(target, self.output_dir, variant=variant)
                self.assertEqual(captured["reference"], expected)
                self.assertEqual(captured["filename"], expected.rsplit(":", 1)[1])
                self.assertEqual(
                    captured["destination"], self.output_dir / captured["filename"]
                )
                self.assertEqual(digest, "0" * 64)

    def test_archive_filename_does_not_replace_the_nashe_default(self):
        captured = {}

        def fake_download(asset, destination):
            captured["reference"] = asset.reference
            return "2" * 64

        with mock.patch.object(download_mod, "download_asset", fake_download):
            download_mod.download_target(
                "s100",
                self.output_dir,
                variant="152",
                filename="s100/resnet152_224x224_nv12.hbm",
            )
        self.assertEqual(
            captured["reference"],
            "s:resnet:s100/resnet152_224x224_nv12.hbm",
        )
        with mock.patch.object(download_mod, "download_asset", fake_download):
            download_mod.download_target("s100", self.output_dir, variant="152")
        self.assertEqual(
            captured["reference"],
            "s:resnet:s100/resnet152_nashe_224x224_nv12.hbm",
        )


if __name__ == "__main__":
    unittest.main()
