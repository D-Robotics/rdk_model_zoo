from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[4]
CONVERSION = Path(__file__).resolve().parents[1] / "conversion"
SOURCE_MAP = CONVERSION / "SOURCE_MAP.json"
ANCHORS = ("source-model", "toolchain-targets", "export", "calibration", "compile", "validation", "artifacts", "known-gaps")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_script(name: str):
    path = CONVERSION / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"efficient_sam_conversion_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ConversionLayoutTest(unittest.TestCase):
    def test_source_map_json_and_source_hashes(self):
        data = json.loads(SOURCE_MAP.read_text())
        self.assertEqual(data["sample"], "efficient_sam")
        mapped = set()
        for entry in data["entries"]:
            unified = CONVERSION / entry["unified"]
            self.assertTrue(unified.is_file(), entry["unified"])
            for source in entry["sources"]:
                path = ROOT / source["path"]
                self.assertTrue(path.is_file(), source["path"])
                self.assertEqual(sha256(path), source["sha256"], source["path"])
                mapped.add(path)
                if entry["kind"] == "verbatim":
                    self.assertEqual(sha256(unified), source["sha256"], entry["unified"])
                if "unified_sha256" in source:
                    self.assertEqual(sha256(unified), source["unified_sha256"], entry["unified"])
        for source in data["source_only"]:
            path = ROOT / source["path"]
            self.assertTrue(path.is_file(), source["path"])
            self.assertEqual(sha256(path), source["sha256"], source["path"])
            mapped.add(path)
        source_root = {p for p in (ROOT / "platforms/x5/samples/vision/efficient_sam/conversion").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        source_root |= {p for p in (ROOT / "platforms/s/samples/vision/efficient_sam/conversion").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        self.assertEqual(mapped, source_root)

    def test_bilingual_contract_anchors_match(self):
        en = (CONVERSION / "README.md").read_text()
        cn = (CONVERSION / "README_cn.md").read_text()
        for anchor in ANCHORS:
            self.assertIn(f'id="{anchor}"', en)
            self.assertIn(f'id="{anchor}"', cn)

    def test_target_configs_preserve_recipe_signals(self):
        for target, march in (("s100", "nash-e"), ("s100p", "nash-m"), ("s600", "nash-p")):
            for role in ("encoder", "decoder"):
                text = next((CONVERSION / "configs" / target).glob(f"*{role}*.yaml")).read_text()
                self.assertIn(f"march: {march}", text)
                self.assertIn("set_all_nodes_int16", text)
                self.assertIn("calibration_type: 'max'", text)
        x5 = (CONVERSION / "configs/x5/efficient_sam_vitt_encoder_featuremap_config.yaml").read_text()
        self.assertIn("march: 'bayes-e'", x5)
        self.assertIn("calibration_type: 'default'", x5)

    def test_all_conversion_parsers_are_importable_without_running(self):
        for name in ("download_assets.py", "dump_encoder_embedding.py", "export_encoder_onnx.py", "export_decoder_onnx.py", "prepare_calibration.py", "prepare_efficient_decoder_calibration.py", "quantize.py"):
            module = load_script(name)
            parser = module.build_parser()
            argv = [] if name == "dump_encoder_embedding.py" else ["--target", "s100"]
            if name == "prepare_calibration.py":
                argv += ["--src", ".", "--out", "./tmp-calibration"]
            elif name == "prepare_efficient_decoder_calibration.py":
                argv += ["--embedding", "./embedding.bin"]
            elif name == "dump_encoder_embedding.py":
                argv += ["--image", "./image.jpg"]
            args = parser.parse_args(argv)
            if name != "dump_encoder_embedding.py":
                self.assertEqual(args.target, "s100")

    def test_export_uses_registered_encoder_module_wrapper(self):
        text = (CONVERSION / "scripts/export_encoder_onnx.py").read_text()
        self.assertIn("class EfficientSAMImageEncoder(torch.nn.Module)", text)
        self.assertIn("self.model = wrapped_model", text)
        self.assertIn("encoder = _build_encoder_wrapper(model, torch)", text)
        self.assertIn("torch.onnx.export(\n            encoder,", text)

    def test_encoder_wrapper_fixture_registers_model_and_delegates(self):
        module = load_script("export_encoder_onnx.py")

        class FakeModule:
            def __init__(self):
                self.training = True

            def eval(self):
                self.training = False
                return self

        class FakeTorch:
            class nn:
                Module = FakeModule

        class Model:
            def get_image_embeddings(self, tensor):
                return ("embedding", tensor)

        wrapped = module._build_encoder_wrapper(Model(), FakeTorch)
        self.assertIsInstance(wrapped, FakeModule)
        self.assertFalse(wrapped.training)
        self.assertEqual(wrapped.forward("input"), ("embedding", "input"))
        self.assertIsInstance(wrapped.model, Model)

    def test_download_helper_mocks_clone_zip_and_checkpoint(self):
        module = load_script("download_assets.py")
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            calls = []

            def fake_retrieve(url, destination):
                calls.append(url)
                if url == "zip://fixture":
                    with zipfile.ZipFile(destination, "w") as archive:
                        archive.writestr("EfficientSAM-main/README.md", "fixture")
                else:
                    Path(destination).write_bytes(b"checkpoint")

            with mock.patch.object(module, "run", side_effect=subprocess_clone_failure), mock.patch.object(module, "urlretrieve", side_effect=fake_retrieve):
                module.main(["--target", "x5", "--workspace", str(workspace), "--zip-url", "zip://fixture", "--checkpoint-url", "checkpoint://fixture"])
            self.assertTrue((workspace / "EfficientSAM/README.md").is_file())
            self.assertTrue((workspace / "EfficientSAM/weights/efficient_sam_vitt.pt").is_file())
            self.assertEqual(calls, ["zip://fixture", "checkpoint://fixture"])

    def test_download_helper_rejects_zip_path_escape(self):
        module = load_script("download_assets.py")
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            archive = workspace / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../outside.txt", "bad")
            with self.assertRaises(RuntimeError):
                module._safe_extract(archive, workspace)

    def test_download_helper_does_not_touch_existing_checkout(self):
        module = load_script("download_assets.py")
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            repo = workspace / "EfficientSAM"
            repo.mkdir()
            marker = repo / "user-file.txt"
            marker.write_text("keep")

            def fake_retrieve(url, destination):
                Path(destination).write_bytes(b"checkpoint")

            with mock.patch.object(module, "run") as clone, mock.patch.object(module, "urlretrieve", side_effect=fake_retrieve):
                module.main(["--target", "x5", "--workspace", str(workspace), "--checkpoint-url", "checkpoint://fixture"])
            clone.assert_not_called()
            self.assertEqual(marker.read_text(), "keep")

    def test_x5_export_rejects_custom_checkpoint_before_torch_import(self):
        with tempfile.TemporaryDirectory() as directory:
            module = load_script("export_encoder_onnx.py")
            args = module.build_parser().parse_args(["--target", "x5", "--repo", directory, "--checkpoint", str(Path(directory) / "custom.pt")])
            with self.assertRaises(ValueError):
                module._load_model(args)

    def test_no_generated_conversion_artifacts_are_checked_in(self):
        forbidden_suffixes = {".onnx", ".bin", ".hbm", ".npy", ".rgbchw", ".json"}
        for path in CONVERSION.rglob("*"):
            if path == SOURCE_MAP:
                continue
            self.assertNotIn(path.suffix, forbidden_suffixes, path)
        for name in ("workspace", "calibration_data_rgbchw_512", "decoder_calibration", "bpu_model_output_512_default_none"):
            self.assertFalse((CONVERSION / name).exists(), name)

def subprocess_clone_failure(*args, **kwargs):
    raise RuntimeError("mock clone failure")


if __name__ == "__main__":
    unittest.main()
