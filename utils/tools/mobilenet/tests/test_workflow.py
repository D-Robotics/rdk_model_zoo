"""Host-only MobileNet workflow integration tests; no weights or SDK needed."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from timm.data import create_transform

from utils.py_utils.classification_host import float_input, prepare_rgb, sha256_file
from utils.tools.mobilenet.workflow import PINS, make_config, prepare_calibration, verify_export

SPECS = json.loads(PINS.read_text())["models"]


class WorkflowTests(unittest.TestCase):
    """Check preprocess parity and graph/calibration evidence coupling."""

    def test_all_contracts_match_timm_transform(self):
        rng = np.random.default_rng(42)
        for key, spec in SPECS.items():
            contract = spec["contract"]
            reference = create_transform(input_size=(3, contract["size"], contract["size"]),
                                         is_training=False, interpolation="bicubic",
                                         crop_pct=contract["crop_pct"], mean=contract["mean"],
                                         std=contract["std"], crop_mode="center")
            for height, width in ((311, 517), (517, 310), (257, 257), (37, 61)):
                image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
                with self.subTest(key=key, shape=(height, width)):
                    actual = float_input(prepare_rgb(image, contract), contract)
                    np.testing.assert_allclose(actual[0], reference(image).numpy(), rtol=0, atol=1e-6)

    def test_config_rejects_cross_model_and_tampered_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = root / "model.onnx"
            graph.write_bytes(b"fixture")
            meta = {"status": "passed", "model": "v4-small", "spec": SPECS["v4-small"],
                    "onnx_sha256": sha256_file(graph)}
            (root / "export.json").write_text(json.dumps(meta))
            verify_export(root, "v4-small", SPECS["v4-small"])
            with self.assertRaises(ValueError):
                verify_export(root, "v2-100", SPECS["v2-100"])
            graph.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_export(root, "v4-small", SPECS["v4-small"])

    def test_calibration_overlap_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (224, 224)).save(image)
            manifest = {"count": 1, "images": [{"path": "image.png", "sha256": sha256_file(image)}]}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            args = argparse.Namespace(model="v4-small", platform="x5", manifest=path,
                                      images_root=root, expected_images=1, evaluation_manifest=path,
                                      output=root / "output")
            with self.assertRaisesRegex(ValueError, "overlap"):
                prepare_calibration(args, SPECS["v4-small"])
            self.assertFalse(args.output.exists())

    def test_config_platform_and_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = SPECS["v4-small"]
            (root / "model.onnx").write_bytes(b"fixture")
            (root / "export.json").write_text(json.dumps({"status": "passed", "model": "v4-small",
                "spec": spec, "onnx_sha256": sha256_file(root / "model.onnx")}))
            (root / "data").mkdir()
            np.save(root / "data/000.npy", np.zeros((1, 3, 224, 224), dtype=np.float32))
            metadata = {"status": "prepared", "model": "v4-small", "spec": spec,
                        "platform": "s100p", "outputs": [{"file": "000.npy", "sha256": sha256_file(root / "data/000.npy")}]}
            (root / "calibration.json").write_text(json.dumps(metadata))
            args = argparse.Namespace(model="v4-small", platform="s100p", export_dir=root,
                                      calibration_dir=root, output=root / "config")
            make_config(args, spec)
            result = yaml.safe_load((args.output / "config.yaml").read_text())
            self.assertEqual(result["model_parameters"]["march"], "nash-m")
            self.assertEqual(result["input_parameters"]["input_type_train"], "rgb")
            mean = np.array(result["input_parameters"]["mean_value"].split(), dtype=float)
            scale = np.array(result["input_parameters"]["scale_value"].split(), dtype=float)
            np.testing.assert_allclose((255 - mean) * scale,
                                       (1 - np.array(spec["contract"]["mean"])) / spec["contract"]["std"])
            args.platform = "x5"
            args.output = root / "bad-config"
            with self.assertRaises(ValueError):
                make_config(args, spec)

    def test_x5_calibration_has_no_numpy_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (224, 224), (10, 20, 30)).save(image)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"count": 1, "dataset": "fixture", "images": [
                {"path": "image.png", "sha256": sha256_file(image)}]}))
            evaluation = root / "evaluation.json"
            evaluation.write_text(json.dumps({"count": 1, "images": [{"sha256": "0" * 64}]}))
            args = argparse.Namespace(model="v4-small", platform="x5", manifest=manifest,
                                      images_root=root, expected_images=1, evaluation_manifest=evaluation,
                                      output=root / "output")
            prepare_calibration(args, SPECS["v4-small"])
            path = args.output / "data/000000.rgb"
            self.assertEqual(path.stat().st_size, 1 * 3 * 224 * 224 * 4)
            np.testing.assert_array_equal(np.fromfile(path, dtype="<f4").reshape(1, 3, 224, 224)[0, :, 0, 0],
                                          [10, 20, 30])


if __name__ == "__main__":
    unittest.main()
