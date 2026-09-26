"""Host conversion preparation with fake compiler, never real OE evidence."""

import contextlib, io, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import yaml
from samples.vision.lanenet.conversion import prepare_calibration, compile as compiler
from samples.vision.lanenet.runtime.python.lanenet import LaneNetTask
from test_lanenet import metadata
from samples.vision.lanenet.runtime.python.model_binding import (
    bind_model,
    resolve_selection,
)


class ConversionTests(unittest.TestCase):
    def prepare(self, root):
        images = root / "images"
        images.mkdir()
        image = np.random.default_rng(4).integers(0, 256, (31, 61, 3), np.uint8)
        cv2.imwrite(str(images / "input.png"), image)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                prepare_calibration.main(
                    [
                        "--images",
                        str(images),
                        "--count",
                        "1",
                        "--output",
                        str(root / "cal"),
                    ]
                ),
                0,
            )
        return image, root / "cal/manifest.json"

    def test_tensor_exact_runtime_and_source_images_bound(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            image, manifest = self.prepare(root)
            j = json.loads(manifest.read_text())
            record = j["records"][0]
            expected = LaneNetTask(
                None, bind_model(resolve_selection("s100"), metadata())
            ).pre_process(image)["input"]
            np.testing.assert_array_equal(
                np.load(root / "cal" / record["tensor"]), expected
            )
            self.assertEqual(len(record["image_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                prepare_calibration.main(
                    [
                        "--images",
                        str(root / "images"),
                        "--count",
                        "1",
                        "--output",
                        str(root / "cal"),
                    ]
                )

    def test_compile_prepare_and_fake_execution_require_real_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, manifest = self.prepare(root)
            onnx = root / "caller.onnx"
            onnx.write_bytes(b"not a real ONNX; host fixture")
            args = [
                "--onnx",
                str(onnx),
                "--calibration-manifest",
                str(manifest),
                "--output",
                str(root / "prepared"),
            ]
            with patch.object(
                compiler.subprocess, "run"
            ) as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compiler.main([*args, "--prepare-only"]), 0)
                run.assert_not_called()
            config = yaml.safe_load((root / "prepared/config.yaml").read_text())
            self.assertEqual(config["input_parameters"]["input_type_rt"], "featuremap")
            self.assertEqual(config["model_parameters"]["march"], "nash-e")
            self.assertEqual(
                config["model_parameters"]["onnx_model"], str(onnx.resolve())
            )
            with patch.object(compiler.subprocess, "run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = "fixture"
                run.return_value.stderr = ""
                with self.assertRaisesRegex(ValueError, "artifact"):
                    compiler.main([*args[:-1], str(root / "missing")])

            def fake_run(argv, **kwargs):
                cfg = yaml.safe_load(Path(argv[-1]).read_text())
                p = Path(cfg["model_parameters"]["working_dir"])
                p.mkdir(parents=True)
                (p / "lanenet256x512.hbm").write_bytes(b"fake compiler output")
                from subprocess import CompletedProcess

                return CompletedProcess(argv, 0, "fixture", "")

            with patch.object(
                compiler.subprocess, "run", side_effect=fake_run
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compiler.main([*args[:-1], str(root / "compiled")]), 0)
            self.assertEqual(
                json.loads((root / "compiled/report.json").read_text())[
                    "artifact_origin"
                ],
                "caller-converted; not published asset authentication",
            )

    def test_tamper_extra_files_and_invalid_counts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, manifest = self.prepare(root)
            j = json.loads(manifest.read_text())
            tensor = root / "cal" / j["records"][0]["tensor"]
            extra = root / "cal/data/unmanifested.npy"
            np.save(extra, np.zeros((1, 3, 256, 512), np.float32))
            with self.assertRaisesRegex(ValueError, "unmanifested"):
                compiler.validate_calibration(manifest)
            extra.unlink()
            tensor.write_bytes(b"tampered")
            onnx = root / "model.onnx"
            onnx.write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                compiler.main(
                    [
                        "--onnx",
                        str(onnx),
                        "--calibration-manifest",
                        str(manifest),
                        "--output",
                        str(root / "out"),
                        "--prepare-only",
                    ]
                )
            for count in ("0", "-1", "2"):
                with self.assertRaises(ValueError):
                    prepare_calibration.main(
                        [
                            "--images",
                            str(root / "images"),
                            "--count",
                            count,
                            "--output",
                            str(root / "other"),
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
