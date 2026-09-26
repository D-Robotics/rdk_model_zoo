"""Every manifest S100/S100P classifier must remain selectable, not silently skipped."""

from pathlib import Path
import contextlib, io, sys, unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "samples/vision/ultralytics_yolo/runtime/python"))
from yolo_assets import model_filename, manifest_asset
from yolo_platform import resolve_platform
import yolo_download
from unittest.mock import patch


class ClassifierManifestTests(unittest.TestCase):
    def test_all_twenty_s100_s100p_classifiers_resolve_exact_manifest(self):
        for target in ("s100", "s100p"):
            profile = resolve_platform(target)
            for family in ("yolov8", "yolo11"):
                for size in "nsmlx":
                    with self.subTest(target=target, family=family, size=size):
                        asset = manifest_asset(profile, family, "cls", size)
                        self.assertIn("_640x640_", asset.filename)
                        self.assertTrue(
                            asset.url.endswith(
                                asset.filename.replace("_640x640_", "_224x224_")
                            )
                        )
                        self.assertEqual(
                            model_filename(profile, family, "cls", size),
                            Path(asset.filename).name,
                        )
        self.assertIn(
            "_640x640_", model_filename(resolve_platform("x5"), "yolo11", "cls", "n")
        )

    def test_public_downloader_dry_run_resolves_s100_classifier_without_network(self):
        with contextlib.redirect_stdout(io.StringIO()) as out, patch.object(
            sys,
            "argv",
            [
                "yolo_download.py",
                "--platform",
                "s100",
                "--family",
                "yolo11",
                "--task",
                "cls",
                "--model-size",
                "n",
                "--dry-run",
            ],
        ):
            rc = yolo_download.main()
        self.assertIn(rc, (None, 0))
        self.assertIn("yolo11n_cls_nashe_224x224_nv12.hbm", out.getvalue())


if __name__ == "__main__":
    unittest.main()
