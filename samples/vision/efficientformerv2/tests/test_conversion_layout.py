"""Provenance and cross-consistency tests for the EfficientFormerV2 conversion set.

The conversion directory is a verbatim rdk_x5 @ac11571 delivery: the three
``EfficientFormerv2_s{0,1,2}_config.yaml`` reference PTQ configs (march
``bayes-e``, calibration ``max``).  Adjusting a pinned file is a
source-branch change, not a local edit.  Unlike the sibling X5 deliveries,
this one's YAML output prefixes **do** reproduce the manifest basenames;
the anchors below pin that agreement plus the retained source-recipe gaps
(no ONNX exporter, no calibration producer, per-variant ``working_dir``
spellings, and S0-only ``debug_mode``/``set_all_nodes_int16``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import unittest

CONVERSION = Path(__file__).resolve().parents[1] / "conversion"

VERBATIM_SHA256 = {
    # rdk_x5 @ac11571
    "EfficientFormerv2_s0_config.yaml": "a0415f8a4a3f75976be1c8a5a0bf30aa9874b95ee5f61ec10d6ee7147c5d4351",
    "EfficientFormerv2_s1_config.yaml": "530d78e7d2e28eb57832b2f8d48d7d1d8f4de5359b127eac915922be6353fcc9",
    "EfficientFormerv2_s2_config.yaml": "b35d73d6059f5e7765f415eac1863a72792d0e090d6c04640e1164b4814ad699",
}

#: variant -> (onnx name, working_dir, manifest basename, max_percentile).
VARIANT_ROWS = {
    "s0": ("./efficientformerv2_s0.onnx", "EfficientFormerv2_s0_int16_model_output",
           "EfficientFormerv2_s0_224x224_nv12.bin", "0.999"),
    "s1": ("./efficientformerv2_s1.onnx", "EfficientFormerv2_s1_224x224_nv12",
           "EfficientFormerv2_s1_224x224_nv12.bin", "0.999"),
    "s2": ("./efficientformerv2_s2.onnx", "EfficientFormerv2_s2_224x224_nv12",
           "EfficientFormerv2_s2_224x224_nv12.bin", "0.9995"),
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
                    hashlib.sha256(data).hexdigest(), digest,
                    f"{name} drifted from the pinned source bytes",
                )

    def test_conversion_directory_is_exactly_the_source_set(self):
        names = sorted(p.name for p in CONVERSION.iterdir() if p.suffix in {".yaml", ".py"})
        self.assertEqual(
            names, sorted(VERBATIM_SHA256),
            "the conversion set must change only together with its provenance pins",
        )


class RecipeConsistencyTests(unittest.TestCase):
    def test_x5_configs_agree_on_numeric_preprocessing(self):
        for name in VERBATIM_SHA256:
            with self.subTest(file=name):
                text = _yaml_text(name)
                self.assertIn('march: "bayes-e"', text)
                self.assertIn("input_type_rt: 'nv12'", text)
                self.assertIn("input_type_train: 'rgb'", text)
                self.assertIn("input_layout_train: 'NCHW'", text)
                # Same numeric family as the other X5 classification configs.
                self.assertIn("mean_value: 123.675 116.28 103.53", text)
                self.assertIn("scale_value: 0.01712475 0.017507 0.01742919", text)
                self.assertIn("cal_data_dir: './calibration_data_rgb_f32'", text)
                self.assertIn("cal_data_type: 'float32'", text)
                self.assertIn("calibration_type: 'max'", text)

    def test_each_variant_names_its_own_onnx_and_percentile(self):
        for variant, (onnx, _workdir, _basename, percentile) in VARIANT_ROWS.items():
            with self.subTest(variant=variant):
                text = _yaml_text(f"EfficientFormerv2_{variant}_config.yaml")
                self.assertIn(f"onnx_model: '{onnx}'", text)
                self.assertIn(f"max_percentile: {percentile}", text)

    def test_output_prefix_reproduces_manifest_basenames(self):
        """Unlike efficientnet/efficientformer, this delivery's prefixes
        carry the variant identity: prefix + '.bin' == manifest basename,
        and each variant compiles into its own working_dir — no rename or
        collision step is needed.  Pin the agreement."""

        for variant, (_onnx, workdir, basename, _p) in VARIANT_ROWS.items():
            with self.subTest(variant=variant):
                text = _yaml_text(f"EfficientFormerv2_{variant}_config.yaml")
                m = re.search(r"output_model_file_prefix: '([^']+)'", text)
                self.assertIsNotNone(m)
                self.assertEqual(f"{m.group(1)}.bin", basename)
                self.assertIn(f"working_dir: '{workdir}'", text)

    def test_s0_only_quirks_are_scoped_to_s0(self):
        """S0 alone carries ``debug_mode: dump_calibration_data`` and
        ``optimization: set_all_nodes_int16``; S1/S2 do not.  Keep that
        asymmetry visible instead of silently normalizing it."""

        s0 = _yaml_text("EfficientFormerv2_s0_config.yaml")
        self.assertIn('debug_mode: "dump_calibration_data"', s0)
        self.assertIn('optimization: "set_all_nodes_int16"', s0)
        for name in ("EfficientFormerv2_s1_config.yaml", "EfficientFormerv2_s2_config.yaml"):
            text = _yaml_text(name)
            self.assertNotIn("debug_mode", text, name)
            self.assertNotIn("set_all_nodes_int16", text, name)

    def test_no_exporter_or_calibration_producer_is_claimed(self):
        """The X5 delivery ships reference configs only: no exporter script
        for the ONNX inputs and no producer for ``calibration_data_rgb_f32``
        exists in this directory, and the test set must stay that way."""

        py_files = sorted(p.name for p in CONVERSION.glob("*.py"))
        self.assertEqual(py_files, [])
        self.assertFalse(
            (CONVERSION / "calibration_data_rgb_f32").exists(),
            "no producer ships for this directory; checked-in data would "
            "fake a reproducible pipeline",
        )


if __name__ == "__main__":
    unittest.main()
