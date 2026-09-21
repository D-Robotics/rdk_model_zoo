"""Provenance and cross-consistency tests for the EfficientNet conversion set.

The conversion directory mixes two verbatim source deliveries:

- rdk_x5 @ac11571 ``samples/vision/efficientnet/conversion/`` — the three
  ``EfficientNet_B{2,3,4}_config.yaml`` reference PTQ configs (march
  ``bayes-e``);
- rdk_s @380e1a2 ``samples/vision/efficientnet/conversion/`` — the complete
  lite0..lite4 recipe (5 YAMLs + 5 timm exporters + ``timm2onnx_local.py`` +
  ``get_calibration_data.py`` + ``x86_inference.py``, march ``nash-e``).

Adjusting any pinned file is a source-branch change, not a local edit.  The
numeric-consistency test pins the *verified* agreement between the S-side
calibration script and the lite YAMLs (mean 127/127/127, scale 0.007843 per
channel) — unlike the retained resnet152 scale discrepancy, this delivery is
internally consistent, and these anchors keep it that way.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import unittest

CONVERSION = Path(__file__).resolve().parents[1] / "conversion"

VERBATIM_SHA256 = {
    # rdk_x5 @ac11571
    "EfficientNet_B2_config.yaml": "916609a72f5bf655a0895ec25a4ac09fbe7d57c54fa7db17662941fdf31e778d",
    "EfficientNet_B3_config.yaml": "543d8d0c01e3855a528d756c891d934ebcb2e13478b531e2a5eb4c18475e4b5f",
    "EfficientNet_B4_config.yaml": "437d6c61a5538494df75d42087e693e3aef6e27696d57954e4a68efdaf817866",
    # rdk_s @380e1a2
    "efficientnet_lite0_config.yaml": "e1a81e3228677218fa8eafdecfe7029ff22e307672a5c03d77b8661c08b543fe",
    "efficientnet_lite1_config.yaml": "9331da0838fe57af8a19bd11f3d50e00f445eacd2d6a6eda6285d63ce9ce3daa",
    "efficientnet_lite2_config.yaml": "3ae0897906768041827c2f74e1f535936a9dd8b0deb3d2c9955e53fd91cdb714",
    "efficientnet_lite3_config.yaml": "13bd9522a3408661c33ece9f70f92d3f5368ea09317ff662e0e48ab362659c7f",
    "efficientnet_lite4_config.yaml": "ed191e49a4f112a4c26d20019cf23feddf46abb8cc4b705494ea3858e0ffd68d",
    "get_calibration_data.py": "8f337b020588a4305528bbe0a8b2c79a2cbd73f119312507ed51d30b4ae1b912",
    "get_efficientnet_lite0_onnx.py": "0027ffec48b7e6033b4ead85435f2591cf198b6baed2756ffd5125b6b8174a44",
    "get_efficientnet_lite1_onnx.py": "d998f0f749fd4c98d51c6c42dd1b9ce57fea7e51dc3f00b941eafbcd138a4f5d",
    "get_efficientnet_lite2_onnx.py": "3e1ea53d0d8bf6e280177a4fab4617f4134f8231aaea46063370ad94d0dcfe45",
    "get_efficientnet_lite3_onnx.py": "7ddb548bf1046c03c1efabbbf925c5117862c489677ddd55a9980af93ddec9d0",
    "get_efficientnet_lite4_onnx.py": "72654d84bb3e472b1a4b87a769bb08ca5f38048ce2d4e5c0c300189927ef8ab3",
    "timm2onnx_local.py": "a81b3acb019efacb4f67cd64caf2e651ac42d4dcdc775dc6255b3cd20c211e10",
    "x86_inference.py": "9de8afdc27e93e1867772a98cac38cc7792b4a87676e69dd0d2d834929a6d325",
}

#: lite variant -> (onnx name, output prefix, declared input size).
LITE_ROWS = {
    "lite0": ("./tf_efficientnet_lite0.onnx", "efficientnet_lite0_224x224_nv12", 224),
    "lite1": ("./tf_efficientnet_lite1.onnx", "efficientnet_lite1_240x240_nv12", 240),
    "lite2": ("./tf_efficientnet_lite2.onnx", "efficientnet_lite2_260x260_nv12", 260),
    "lite3": ("./tf_efficientnet_lite3.onnx", "efficientnet_lite3_300x300_nv12", 300),
    "lite4": ("./tf_efficientnet_lite4.onnx", "efficientnet_lite4_380x380_nv12", 380),
}

#: Manifest filenames (published identity) for the lite rows; the YAML output
#: prefix must reproduce the manifest basename exactly.
LITE_MANIFEST_BASENAMES = {
    "lite0": "efficientnet_lite0_224x224_nv12.hbm",
    "lite1": "efficientnet_lite1_240x240_nv12.hbm",
    "lite2": "efficientnet_lite2_260x260_nv12.hbm",
    "lite3": "efficientnet_lite3_300x300_nv12.hbm",
    "lite4": "efficientnet_lite4_380x380_nv12.hbm",
}


def _yaml_text(name: str) -> str:
    path = CONVERSION / name
    if not path.is_file():
        raise AssertionError(f"missing conversion file: {name}")
    return path.read_text()


class ProvenanceTests(unittest.TestCase):
    def test_verbatim_files_match_source_sha256(self):
        for name, digest in VERBATIM_SHA256.items():
            with self.subTest(file=name):
                data = (CONVERSION / name).read_bytes()
                self.assertEqual(
                    hashlib.sha256(data).hexdigest(),
                    digest,
                    f"{name} drifted from its pinned source bytes",
                )


class LiteRecipeConsistencyTests(unittest.TestCase):
    def _field(self, text: str, key: str) -> str:
        match = re.search(rf"^\s*{key}:\s*['\"]?([^'\"\n]+?)['\"]?\s*$", text, re.M)
        if not match:
            raise AssertionError(f"key {key!r} not found")
        return match.group(1).strip()

    def test_lite_prefix_matches_manifest_basename(self):
        for variant, (_, prefix, _) in LITE_ROWS.items():
            with self.subTest(variant=variant):
                yaml_name = f"efficientnet_{variant}_config.yaml"
                prefix = self._field(_yaml_text(yaml_name), "output_model_file_prefix")
                self.assertEqual(prefix + ".hbm", LITE_MANIFEST_BASENAMES[variant])

    def test_lite_yamls_agree_on_normalisation_and_target(self):
        for variant in LITE_ROWS:
            with self.subTest(variant=variant):
                text = _yaml_text(f"efficientnet_{variant}_config.yaml")
                self.assertEqual(self._field(text, "march"), "nash-e")
                self.assertEqual(
                    self._field(text, "input_type_rt"), "nv12"
                )
                self.assertEqual(self._field(text, "input_type_train"), "rgb")
                self.assertEqual(self._field(text, "input_layout_train"), "NCHW")
                self.assertEqual(
                    self._field(text, "mean_value"), "127 127 127"
                )
                self.assertEqual(
                    self._field(text, "scale_value"),
                    "0.007843 0.007843 0.007843",
                )
                self.assertEqual(self._field(text, "cal_data_dir"), "./calibration_data_rgb")
                self.assertEqual(self._field(text, "cal_data_type"), "float32")

    def test_calibration_script_matches_lite_yaml_normalisation(self):
        """Verified consistency (unlike resnet152): the script's transform
        chain /255 -> mean 127 -> scale 0.007843 matches the lite YAMLs, so
        the generated calibration data is the declared calibration input."""

        script = _yaml_text("get_calibration_data.py")
        self.assertIn("ScaleTransformer(scale_value=255.0)", script)
        self.assertIn("MeanTransformer(means=np.array([127, 127, 127]))", script)
        self.assertIn(
            "ScaleTransformer(scale_value=np.array([0.007843, 0.007843, 0.007843]))",
            script,
        )
        self.assertIn("output_calib_dir = './calibration_data_rgb/'", script)

    def test_calibration_output_dir_matches_yaml_cal_data_dir(self):
        script = _yaml_text("get_calibration_data.py")
        match = re.search(r"output_calib_dir = '([^']+)'", script)
        self.assertIsNotNone(match)
        out_dir = match.group(1).rstrip("/")
        for variant in LITE_ROWS:
            text = _yaml_text(f"efficientnet_{variant}_config.yaml")
            self.assertEqual(self._field(text, "cal_data_dir"), out_dir)


class X5ConfigFactsTests(unittest.TestCase):
    """Pin the x5-side YAML facts the conversion README states (bayes-e,
    shared normalisation, rgb_f32 calibration dir) plus the retained source
    quirks that the README discloses as gaps instead of hiding."""

    def test_x5_configs_share_declared_normalisation(self):
        for name in ("EfficientNet_B2_config.yaml", "EfficientNet_B3_config.yaml", "EfficientNet_B4_config.yaml"):
            with self.subTest(config=name):
                text = _yaml_text(name)
                self.assertIn("march: \"bayes-e\"", text)
                self.assertIn("mean_value: 123.675 116.28 103.53", text)
                self.assertIn(
                    "scale_value: 0.01712475 0.017507 0.01742919", text
                )
                self.assertIn("input_type_rt: 'nv12'", text)
                self.assertIn("input_type_train: 'rgb'", text)
                self.assertIn("cal_data_dir: './calibration_data_rgb_f32'", text)

    def test_x5_output_prefix_is_variant_less(self):
        """Retained source fact: all three x5 configs emit the same
        variant-less prefix ``EfficientNet_224x224_nv12``, while the
        published manifest filenames carry B2/B3/B4.  The README must keep
        documenting the rename step; this anchor keeps the source quirk
        visible instead of silently 'fixed'."""

        for name in ("EfficientNet_B2_config.yaml", "EfficientNet_B3_config.yaml", "EfficientNet_B4_config.yaml"):
            with self.subTest(config=name):
                text = _yaml_text(name)
                match = re.search(r"output_model_file_prefix: '([^']+)'", text)
                self.assertIsNotNone(match)
                self.assertEqual(match.group(1), "EfficientNet_224x224_nv12")


if __name__ == "__main__":
    unittest.main()
