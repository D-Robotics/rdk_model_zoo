"""Conversion-directory layout and provenance tests (B1-R1 remediation).

The ResNet152 conversion materials are kept byte-verbatim from the rdk_s
source branch; these tests pin that provenance and keep the shipped YAML,
calibration script, and manifest filenames mutually consistent so the
per-variant recipe cannot drift silently.
"""

from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path

from samples._shared.assets import _ROOT, list_assets

CONVERSION_DIR = (
    _ROOT / "samples" / "vision" / "resnet" / "conversion"
)

# rdk_s@380e1a2:samples/vision/resnet152/conversion/ digests (B1-R1
# provenance pins — a mismatch means the file was edited locally instead
# of through the source branch).
SOURCE_DIGESTS = {
    "get_calibration_data.py": (
        "d8a39491c029498906fb1adc633c67d0607ff40fa0c646fe6259d814112e6910"
    ),
    "resnet152_config.yaml": (
        "20eaf2cbf98f1a5079e6f93edc6c95900139c1eb09c69a349bd23020151dd742"
    ),
    "x86_inference.py": (
        "f2c5738e96b90f104bf0a9fad51d531dc533194f6441da620196ba05040a13fa"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _readme_text(name: str) -> str:
    return (CONVERSION_DIR / name).read_text(encoding="utf-8")


class Resnet152ProvenanceTests(unittest.TestCase):
    def test_source_files_kept_verbatim(self) -> None:
        for name, digest in SOURCE_DIGESTS.items():
            with self.subTest(file=name):
                path = CONVERSION_DIR / name
                self.assertTrue(
                    path.is_file(), f"missing conversion file: {name}"
                )
                self.assertEqual(
                    _sha256(path),
                    digest,
                    "file drifted from rdk_s@380e1a2 — source-branch change "
                    "required, not a local edit",
                )

    def test_yaml_matches_calibration_script(self) -> None:
        import yaml

        config = yaml.safe_load(
            (CONVERSION_DIR / "resnet152_config.yaml").read_text("utf-8")
        )
        script = (CONVERSION_DIR / "get_calibration_data.py").read_text("utf-8")

        cal_dir = config["calibration_parameters"]["cal_data_dir"]
        self.assertEqual(cal_dir, "./calibration_data_rgb")
        self.assertIn(
            "output_calib_dir = './calibration_data_rgb/'", script,
            "script output dir must match the YAML cal_data_dir",
        )

        inputs = config["input_parameters"]
        self.assertEqual(inputs["input_type_train"], "rgb")
        self.assertEqual(inputs["input_layout_train"], "NCHW")
        self.assertEqual(inputs["input_type_rt"], "nv12")

        # The script's transformer chain (x255, mean, x0.017) implements
        # the YAML's data_mean_and_scale; the script uses the rounded
        # per-channel values, so compare with a small tolerance.
        means = [float(v) for v in inputs["mean_value"].split()]
        self.assertEqual(means, [123.675, 116.28, 103.53])
        self.assertIn("MeanTransformer(means=np.array([123.675, 116.28, 103.53]))", script)
        scales = [float(v) for v in inputs["scale_value"].split()]
        for value in scales:
            self.assertAlmostEqual(value, 0.017, delta=2e-3)

    def test_output_prefix_matches_manifest_filenames(self) -> None:
        import yaml

        config = yaml.safe_load(
            (CONVERSION_DIR / "resnet152_config.yaml").read_text("utf-8")
        )
        prefix = config["model_parameters"]["output_model_file_prefix"]
        march = config["model_parameters"]["march"]
        self.assertEqual(prefix, "resnet152_224x224_nv12")
        self.assertEqual(march, "nash-e")

        assets = list_assets("s", "resnet152")
        self.assertGreaterEqual(len(assets), 2)
        for asset in assets:
            with self.subTest(filename=asset.filename):
                self.assertEqual(
                    Path(asset.filename).name, f"{prefix}.hbm",
                    "compile output prefix must match the published manifest "
                    "filename",
                )


class VariantCoverageDocTests(unittest.TestCase):
    """Every variant's reproducible scope is stated in both languages."""

    def test_resnet18_flow_still_documented(self) -> None:
        for name in ("README.md", "README_cn.md"):
            with self.subTest(readme=name):
                text = _readme_text(name)
                self.assertIn("export_resnet18_onnx.py", text)
                self.assertIn("13_resnet18", text)

    def test_resnet50_pointer_only_recipe_declared(self) -> None:
        for name in ("README.md", "README_cn.md"):
            with self.subTest(readme=name):
                text = _readme_text(name)
                self.assertIn("13_resnet50", text)
                # The missing in-repo recipe must be declared, not implied.
                self.assertRegex(
                    text,
                    r"(?i)resnet50[^\n]*(no|未|没有)[^\n]*(recipe|配方|YAML|export|导出)",
                )

    def test_resnet152_recipe_documented_with_inputs(self) -> None:
        for name in ("README.md", "README_cn.md"):
            with self.subTest(readme=name):
                text = _readme_text(name)
                self.assertIn("resnet152_config.yaml", text)
                self.assertIn("get_calibration_data.py", text)
                self.assertIn("x86_inference.py", text)
                # User-provided calibration input is called out (the
                # script's default source dir is from the legacy tree).
                self.assertRegex(
                    text, r"(?i)(user-provided|用户自备|user-edited|用户需修改)"
                )
                self.assertIn("resnet152_224x224_nv12.hbm", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
