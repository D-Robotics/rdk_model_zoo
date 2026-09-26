"""Conversion prerequisite and emitted-input checks without Torch or OE."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from samples.vision.yolo26_depth.conversion import prepare_calibration as cal
from samples.vision.yolo26_depth.conversion import compile as compiler
from samples.vision.yolo26_depth.conversion import export
from test_depth import ROOT


class ConversionTests(unittest.TestCase):
    def test_calibration_geometries_and_dtypes(self):
        image = np.arange(37 * 23 * 3, dtype=np.uint8).reshape(37, 23, 3)
        x, gx = cal.prepare_tensor(image, "x5", "nv12")
        s, gs = cal.prepare_tensor(image, "s100", "nv12")
        lite, gl = cal.prepare_tensor(image, "s100", "lite")
        self.assertEqual(x.shape, (3, 768, 768))
        self.assertEqual(x.dtype, np.uint8)
        self.assertEqual(s.shape, (1, 3, 768, 768))
        self.assertEqual(s.dtype, np.float32)
        np.testing.assert_array_equal(s, x[None].astype(np.float32) / 255.0)
        from samples.vision.yolo26_depth.runtime.python.model_binding import (
            bind_model,
            resolve_selection,
        )
        from samples.vision.yolo26_depth.runtime.python.yolo26_depth import (
            Yolo26DepthTask,
        )
        from test_depth import metadata

        task = Yolo26DepthTask(
            None, bind_model(resolve_selection("s100", variant="l"), metadata(True))
        )
        np.testing.assert_array_equal(lite, task.pre_process(image).tensors["images"])
        self.assertEqual(gx, gs)
        self.assertNotEqual(gx, gl)

    def test_calibration_real_files_manifest_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            images = p / "images"
            images.mkdir()
            cv2.imwrite(str(images / "a.png"), np.zeros((20, 31, 3), np.uint8))
            for target, variant in [("x5", "n"), ("s100", "n"), ("s600", "l")]:
                output = p / f"{target}-{variant}"
                with contextlib.redirect_stdout(io.StringIO()):
                    cal.main(
                        [
                            "--target",
                            target,
                            "--variant",
                            variant,
                            "--images",
                            str(images),
                            "--output",
                            str(output),
                            "--count",
                            "1",
                        ]
                    )
                manifest = json.loads(output.with_suffix(".json").read_text())
                self.assertEqual(manifest["target"], target)
                self.assertEqual(len(manifest["records"]), 1)
                record = manifest["records"][0]
                self.assertTrue((output / record["output"]).is_file())
                self.assertEqual(len(record["output_sha256"]), 64)
                with self.assertRaises(FileExistsError):
                    cal.main(
                        [
                            "--target",
                            target,
                            "--variant",
                            variant,
                            "--images",
                            str(images),
                            "--output",
                            str(output),
                            "--count",
                            "1",
                        ]
                    )
            with self.assertRaises(ValueError):
                cal.main(
                    [
                        "--target",
                        "x5",
                        "--images",
                        str(images),
                        "--output",
                        str(p / "invalid"),
                        "--count",
                        "0",
                    ]
                )
            self.assertFalse((p / "invalid").exists())

    def test_all_29_templates_preserved_and_compilation_paths_bound(self):
        for target in ("x5", "s100", "s100p", "s600"):
            for variant in ("n", "s", "m", "l", "x"):
                profiles = (
                    ["nv12"]
                    if target == "x5"
                    else (["nv12", "lite"] if variant in "nsm" else ["lite"])
                )
                for profile in profiles:
                    template = compiler.template_path(target, variant, profile)
                    source = (
                        ROOT
                        / f'platforms/{"x5" if target=="x5" else "s"}/samples/vision/yolo26_depth/conversion/ptq_yamls'
                        / template.name
                    )
                    self.assertEqual(template.read_bytes(), source.read_bytes())
                    config = compiler.configure(
                        target,
                        variant,
                        profile,
                        Path("/tmp/depth.onnx"),
                        Path("/tmp/cal"),
                        Path("/tmp/work"),
                    )
                    self.assertEqual(
                        config["model_parameters"]["onnx_model"],
                        str(Path("/tmp/depth.onnx").resolve()),
                    )
                    self.assertEqual(
                        config["calibration_parameters"]["cal_data_dir"],
                        str(Path("/tmp/cal").resolve()),
                    )
                    self.assertEqual(
                        config["model_parameters"]["working_dir"],
                        str(Path("/tmp/work").resolve()),
                    )

    def test_compile_prepare_only_checks_profile_and_digests(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            images = p / "images"
            images.mkdir()
            cv2.imwrite(str(images / "a.png"), np.zeros((20, 31, 3), np.uint8))
            with contextlib.redirect_stdout(io.StringIO()):
                cal.main(
                    [
                        "--target",
                        "s100",
                        "--images",
                        str(images),
                        "--output",
                        str(p / "cal"),
                        "--count",
                        "1",
                    ]
                )
            onnx = p / "n.onnx"
            onnx.write_bytes(b"fixture, not a parsed ONNX graph")
            args = [
                "--target",
                "s100",
                "--variant",
                "n",
                "--onnx",
                str(onnx),
                "--calibration-manifest",
                str(p / "cal.json"),
            ]
            with patch.object(compiler, "execute") as call, contextlib.redirect_stdout(
                io.StringIO()
            ):
                compiler.main(
                    [*args, "--output", str(p / "prepared"), "--prepare-only"]
                )
                call.assert_not_called()
            report = json.loads((p / "prepared/preparation.json").read_text())
            self.assertEqual(report["compilation"], "not-run")
            self.assertEqual(report["onnx_graph_validation"], "not-run")
            with self.assertRaisesRegex(ValueError, "profile"):
                compiler.main(
                    [
                        *args,
                        "--experimental-lite",
                        "--output",
                        str(p / "wrong"),
                        "--prepare-only",
                    ]
                )
            (p / "cal/0000.npy").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "digest"):
                compiler.main([*args, "--output", str(p / "changed"), "--prepare-only"])

    def test_compiler_exit_success_without_artifact_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            images = p / "images"
            images.mkdir()
            cv2.imwrite(str(images / "a.png"), np.zeros((20, 31, 3), np.uint8))
            with contextlib.redirect_stdout(io.StringIO()):
                cal.main(
                    [
                        "--target",
                        "s100",
                        "--images",
                        str(images),
                        "--output",
                        str(p / "cal"),
                        "--count",
                        "1",
                    ]
                )
            onnx = p / "n.onnx"
            onnx.write_bytes(b"host fixture")
            argv = [
                "--target",
                "s100",
                "--variant",
                "n",
                "--onnx",
                str(onnx),
                "--calibration-manifest",
                str(p / "cal.json"),
                "--output",
                str(p / "compile"),
            ]
            with patch.object(compiler, "execute") as execute:
                with self.assertRaisesRegex(FileNotFoundError, "expected artifact"):
                    compiler.main(argv)
                self.assertEqual(execute.call_args.args[0][0], "hb_compile")
            self.assertFalse((p / "compile" / "compile-report.json").exists())

    def test_fake_compiler_artifact_and_source_metric_collection(self):
        for target in ("x5", "s100"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)
                images = p / "images"
                images.mkdir()
                cv2.imwrite(str(images / "a.png"), np.zeros((20, 31, 3), np.uint8))
                with contextlib.redirect_stdout(io.StringIO()):
                    cal.main(
                        [
                            "--target",
                            target,
                            "--images",
                            str(images),
                            "--output",
                            str(p / "cal"),
                            "--count",
                            "1",
                        ]
                    )
                onnx = p / "n.onnx"
                onnx.write_bytes(b"host fixture")
                commands = []

                def execute(command, cwd, log):
                    commands.append(command)
                    log.write_text(
                        "The quantized model output:\noutput0 0.999\nFPS=10.0, latency = 100000.0 us, DDR = 1024 bytes\n"
                    )
                    if command[0] == "hb_compile" or "makertbin" in command:
                        config = yaml.safe_load(
                            Path(command[command.index("--config") + 1]).read_text()
                        )
                        model = config["model_parameters"]
                        working = Path(model["working_dir"])
                        working.mkdir()
                        ext = ".bin" if target == "x5" else ".hbm"
                        (
                            working / (model["output_model_file_prefix"] + ext)
                        ).write_bytes(b"fake model")
                        if target == "x5":
                            (
                                working
                                / (
                                    model["output_model_file_prefix"]
                                    + "_quantized_model.onnx"
                                )
                            ).write_bytes(b"fake quantized graph")

                with patch.object(
                    compiler, "execute", side_effect=execute
                ), contextlib.redirect_stdout(io.StringIO()):
                    compiler.main(
                        [
                            "--target",
                            target,
                            "--variant",
                            "n",
                            "--onnx",
                            str(onnx),
                            "--calibration-manifest",
                            str(p / "cal.json"),
                            "--output",
                            str(p / "compile"),
                        ]
                    )
                report = json.loads((p / "compile" / "compile-report.json").read_text())
                self.assertTrue(Path(report["artifact"]).is_file())
                self.assertEqual(report["board"], "not-run")
                if target == "x5":
                    self.assertEqual(report["output_cosine_similarity"], 0.999)
                    self.assertEqual(len(commands), 3)
                else:
                    self.assertEqual(len(commands), 1)

    def test_export_help_is_host_only_and_boundary_selection(self):
        path = ROOT / "samples/vision/yolo26_depth/conversion/export.py"
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            capture_output=True,
            text=True,
            cwd="/tmp",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(export.resolve_boundary("x5", "l", None), "log")
        self.assertEqual(export.resolve_boundary("s100", "l", None), "lite")
        self.assertEqual(export.resolve_boundary("s100", "n", None), "log")
        with self.assertRaises(ValueError):
            export.resolve_boundary("x5", "n", "lite")
        self.assertEqual(
            export.export_name("n", 11, "log"), "yolo26n-depth_op11_log.onnx"
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                export.main(
                    [
                        "--target",
                        "x5",
                        "--weights",
                        str(p / "missing.pt"),
                        "--variant",
                        "n",
                        "--output-dir",
                        str(p / "out"),
                    ]
                )
            self.assertFalse((p / "out").exists())


if __name__ == "__main__":
    unittest.main()
