"""Source sample identities must not be silently replaced by family assets."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
PYTHON = ROOT / "samples/vision/ultralytics_yolo/runtime/python"
sys.path.insert(0, str(PYTHON))
from samples._shared.assets import list_assets
from yolo_platform import resolve_platform
from yolo_assets import UnsupportedAssetError, model_path
import yolo_download

spec = importlib.util.spec_from_file_location("standalone_cli_test", PYTHON / "main.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

SAMPLES = {
    "yolo11": ("yolo11", "detect", 0.45),
    "yolo11_pose": ("yolo11", "pose", 0.7),
    "yolo11_seg": ("yolo11", "seg", 0.7),
    "yolov13_imoonlab": ("yolov13", "detect", 0.45),
}


class StandaloneAssets(unittest.TestCase):
    def test_all_ten_source_assets_keep_identity_url_and_isolated_path(self):
        count = 0
        for sample, (family, task, nms) in SAMPLES.items():
            for asset in list_assets("s", sample):
                target = asset.filename.split("/")[0]
                with self.subTest(asset=asset.reference):
                    args = cli.build_parser().parse_args(
                        [
                            "--platform",
                            target,
                            "--task",
                            task,
                            "--asset-id",
                            asset.reference,
                        ]
                    )
                    plan = cli.describe_plan(resolve_platform(target), args)
                    self.assertEqual(plan["asset_reference"], asset.reference)
                    self.assertEqual(plan["url"], asset.url)
                    self.assertTrue(
                        plan["path"].endswith(f"/standalone/{sample}/{asset.filename}")
                    )
                    self.assertEqual(args.family, family)
                    self.assertEqual(args.nms_thres, nms)
                    count += 1
        self.assertEqual(count, 10)

    def test_conflicting_target_task_family_size_reject_without_fallback(self):
        ref = "s:yolo11_pose:s100/yolo11n_pose_nashe_640x640_nv12.hbm"
        for extra in [
            ["--platform", "s100p", "--task", "pose"],
            ["--platform", "s600", "--task", "pose"],
            ["--platform", "s100", "--task", "detect"],
            ["--platform", "s100", "--task", "pose", "--family", "yolov8"],
            ["--platform", "s100", "--task", "pose", "--model-size", "s"],
        ]:
            with self.subTest(extra=extra):
                args = cli.build_parser().parse_args(extra + ["--asset-id", ref])
                with self.assertRaises(UnsupportedAssetError):
                    cli.describe_plan(resolve_platform(args.platform), args)

    def test_s100_yolov13_family_resolves_source_default_and_other_targets_reject(self):
        args = cli.build_parser().parse_args(
            ["--platform", "s100", "--family", "yolov13"]
        )
        plan = cli.describe_plan(resolve_platform("s100"), args)
        self.assertEqual(
            plan["asset_reference"],
            "s:yolov13_imoonlab:s100/yolo13n_detect_nashe_640x640_nv12.hbm",
        )
        self.assertIn("/standalone/yolov13_imoonlab/s100/", plan["path"])
        self.assertEqual(
            model_path(
                str(PYTHON.parents[1] / "model"),
                resolve_platform("s100"),
                "yolov13",
                "detect",
            ),
            plan["path"],
        )
        for target in ["s100p", "s600"]:
            args = cli.build_parser().parse_args(
                ["--platform", target, "--family", "yolov13"]
            )
            with self.assertRaises(UnsupportedAssetError):
                cli.describe_plan(resolve_platform(target), args)

    def test_explicit_path_and_threshold_keep_caller_values(self):
        args = cli.build_parser().parse_args(
            [
                "--platform",
                "s100",
                "--task",
                "seg",
                "--asset-id",
                "s:yolo11_seg:s100/yolo11n_seg_nashe_640x640_nv12.hbm",
                "--model-path",
                "/external/custom.hbm",
                "--nms-thres",
                ".6",
            ]
        )
        plan = cli.describe_plan(resolve_platform("s100"), args)
        self.assertEqual(plan["path"], "/external/custom.hbm")
        self.assertIsNone(plan["url"])
        self.assertEqual(args.nms_thres, 0.6)

    def test_exact_downloader_dry_run_uses_same_path_and_never_downloads(self):
        ref = "s:yolo11_seg:s600/yolo11n_seg_nashe_640x640_nv12.hbm"
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(
            io.StringIO()
        ) as out, patch.object(
            sys,
            "argv",
            [
                "download",
                "--platform",
                "s600",
                "--asset-id",
                ref,
                "--model-dir",
                directory,
                "--dry-run",
            ],
        ), patch(
            "samples._shared.assets.download_asset"
        ) as download:
            self.assertEqual(yolo_download.main(), 0)
            download.assert_not_called()
            self.assertIn(
                "/standalone/yolo11_seg/s600/yolo11n_seg_nashe_640x640_nv12.hbm",
                out.getvalue(),
            )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_downloader_rejects_asset_id_with_all_and_conflicting_task(self):
        ref = "s:yolo11_pose:s100/yolo11n_pose_nashe_640x640_nv12.hbm"
        for extra in [["--all"], ["--task", "detect"]]:
            with contextlib.redirect_stderr(io.StringIO()), patch.object(
                sys,
                "argv",
                ["download", "--platform", "s100", "--asset-id", ref, "--dry-run"]
                + extra,
            ):
                self.assertEqual(yolo_download.main(), 2)

    def test_real_preparation_path_passes_the_exact_asset_to_shared_downloader(self):
        asset = list_assets("s", "yolo11")[0]
        target = asset.filename.split("/")[0]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(
            io.StringIO()
        ), patch.object(
            sys,
            "argv",
            [
                "download",
                "--platform",
                target,
                "--asset-id",
                asset.reference,
                "--model-dir",
                directory,
            ],
        ), patch(
            "samples._shared.assets.download_asset"
        ) as download:
            self.assertEqual(yolo_download.main(), 0)
            passed_asset, path = download.call_args.args
            self.assertEqual(passed_asset, asset)
            self.assertEqual(
                path, Path(directory) / "standalone" / "yolo11" / asset.filename
            )
            self.assertEqual(download.call_count, 1)

    def test_same_basename_family_and_source_assets_never_share_destination(self):
        common = ["--platform", "s100", "--task", "pose"]
        source = cli.build_parser().parse_args(
            common
            + ["--asset-id", "s:yolo11_pose:s100/yolo11n_pose_nashe_640x640_nv12.hbm"]
        )
        family = cli.build_parser().parse_args(common + ["--family", "yolo11"])
        a = cli.describe_plan(resolve_platform("s100"), source)
        b = cli.describe_plan(resolve_platform("s100"), family)
        self.assertEqual(a["filename"], b["filename"])
        self.assertNotEqual(a["path"], b["path"])
        self.assertNotEqual(a["url"], b["url"])
        self.assertEqual(source.nms_thres, 0.7)
        self.assertIsNone(family.nms_thres)

    def test_canonical_all_adds_imoonlab_but_legacy_inventory_stays_unchanged(self):
        profile = resolve_platform("s100")
        parser = yolo_download.build_parser()
        canonical = yolo_download.select_assets(parser.parse_args(["--all"]), profile)
        legacy = yolo_download.select_assets(parser.parse_args(["--all", "--legacy-families"]), profile)
        self.assertEqual([item for item in canonical if item[0] == "yolov13"],
                         [("yolov13", "detect", size) for size in "nslx"])
        self.assertFalse(any(item[0] in ("yolo26", "yolov13") for item in legacy))

    def test_list_includes_separate_source_identity(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            cli.print_model_listing(resolve_platform("s100"))
        self.assertIn(
            "s:yolo11_pose:s100/yolo11n_pose_nashe_640x640_nv12.hbm", out.getvalue()
        )
