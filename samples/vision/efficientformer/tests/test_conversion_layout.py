"""Provenance and cross-consistency tests for the EfficientFormer conversion set.

The conversion directory is a verbatim rdk_x5 @ac11571 delivery: the two
``EfficientFormer_l{1,3}_config.yaml`` reference PTQ configs (march
``bayes-e``, calibration ``set_all_nodes_int16``).  Adjusting a pinned file
is a source-branch change, not a local edit.  The anchors below also pin the
*retained* source-recipe gaps: the variant-less output prefix (both YAMLs
emit ``EfficientFormer_224x224_nv12``, which does not match the manifest
basenames) and the absent ONNX/calibration producers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import unittest

CONVERSION = Path(__file__).resolve().parents[1] / "conversion"

VERBATIM_SHA256 = {
    # rdk_x5 @ac11571
    "EfficientFormer_l1_config.yaml": "155d7e9fec3bf5a6ce349eefa27bd194aa2e78656f810d05cefd7d3ed74c1ca8",
    "EfficientFormer_l3_config.yaml": "45107b556548af8299f544869d98c428b6c8ac2e108b3c7f18b7952425764847",
}

#: variant -> (onnx name, working_dir, manifest basename).
VARIANT_ROWS = {
    "l1": ("./efficientformer_l1.onnx", "EfficientFormer_224x224_nv12_int16",
           "EfficientFormer_l1_224x224_nv12.bin"),
    "l3": ("./efficientformer_l3.onnx", "EfficientFormer_224x224_nv12_int16",
           "EfficientFormer_l3_224x224_nv12.bin"),
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
                # Distinctive to this delivery: whole-graph int16 placement.
                self.assertIn('optimization: "set_all_nodes_int16"', text)

    def test_each_variant_names_its_own_onnx(self):
        for variant, (onnx, _workdir, _basename) in VARIANT_ROWS.items():
            with self.subTest(variant=variant):
                text = _yaml_text(f"EfficientFormer_{variant}_config.yaml")
                self.assertIn(f"onnx_model: '{onnx}'", text)

    def test_variant_less_prefix_does_not_reproduce_manifest_basenames(self):
        """Retained source gap: both YAMLs emit the same variant-less prefix,
        so the compiled output must be renamed per variant to match the
        manifest.  This anchor keeps the gap (and the required rename)
        visible instead of letting the prefix silently match."""

        for variant, (_onnx, workdir, basename) in VARIANT_ROWS.items():
            with self.subTest(variant=variant):
                text = _yaml_text(f"EfficientFormer_{variant}_config.yaml")
                m = re.search(r"output_model_file_prefix: '([^']+)'", text)
                self.assertIsNotNone(m)
                prefix = m.group(1)
                self.assertEqual(prefix, "EfficientFormer_224x224_nv12")
                # The prefix is NOT the manifest basename (no l1/l3 infix).
                self.assertNotEqual(f"{prefix}.bin", basename)
                self.assertIn(f"working_dir: '{workdir}'", text)

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
