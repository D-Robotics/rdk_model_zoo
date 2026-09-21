"""Provenance and cross-consistency tests for the EdgeNeXt conversion set.

The conversion directory is a verbatim rdk_x5 @ac11571 delivery: four
reference PTQ configs (base/small/x_small/xx_small, march ``bayes-e``,
calibration ``max`` at percentile 0.999).  Unlike most siblings in this
family, the EdgeNeXt configs are **positively anchored**: each YAML's
``output_model_file_prefix`` reproduces its manifest basename exactly, so no
rename step is needed — these tests pin that agreement so a future drift is
caught.  No exporter script and no ``calibration_data_rgb_f32`` producer ship
with the delivery.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import unittest

CONVERSION = Path(__file__).resolve().parents[1] / "conversion"

VERBATIM_SHA256 = {
    # rdk_x5 @ac11571
    "EdgeNeXt_base_config.yaml": "3b9ae261068014200f0e202ef952edc01370255481fccea26adf4c8987cf017f",
    "EdgeNeXt_small_config.yaml": "aa10bf4d8ef82e5c2229a5afee77fbc6dfbc6e9cee02bc65204203e08aca05a1",
    "EdgeNeXt_x_small_config.yaml": "0c82bd00183efe65ec5a36886540d66d54d67563b9df06a95ce0c03026916931",
    "EdgeNeXt_xx_small_config.yaml": "ace37ce0adcee872a0ba2c24e113c8f43d848e9ac1aa143a56e5a632ad45f62a",
}

#: variant -> (onnx input, emitted prefix == manifest basename)
VARIANTS = {
    "base": ("./edgenext_base.onnx", "EdgeNeXt_base_224x224_nv12.bin"),
    "small": ("./edgenext_small.onnx", "EdgeNeXt_small_224x224_nv12.bin"),
    "x_small": ("./edgenext_x_small.onnx", "EdgeNeXt_x_small_224x224_nv12.bin"),
    "xx_small": ("./edgenext_xx_small.onnx", "EdgeNeXt_xx_small_224x224_nv12.bin"),
}

#: int16 node placements per config, counted from the pinned source bytes
#: (the xx-small model places 16 nodes; the others 3).
INT16_PLACEMENTS = {
    "base": 3, "small": 3, "x_small": 3, "xx_small": 16,
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
    def test_all_configs_declare_the_classification_numeric_family(self):
        for name in VERBATIM_SHA256:
            with self.subTest(file=name):
                text = _yaml_text(name)
                self.assertIn('march: "bayes-e"', text)
                self.assertIn("input_type_rt: 'nv12'", text)
                self.assertIn("input_type_train: 'rgb'", text)
                self.assertIn("input_layout_train: 'NCHW'", text)
                self.assertIn("mean_value: 123.675 116.28 103.53", text)
                self.assertIn("scale_value: 0.01712475 0.017507 0.01742919", text)
                self.assertIn("cal_data_dir: './calibration_data_rgb_f32'", text)
                self.assertIn("cal_data_type: 'float32'", text)
                self.assertIn("calibration_type: 'max'", text)
                self.assertIn("max_percentile: 0.999", text)

    def test_output_prefix_reproduces_the_manifest_basename(self):
        """Positive anchor (the efficientformerv2 form): each YAML's
        ``output_model_file_prefix + '.bin'`` equals its manifest basename,
        and the onnx input name carries the same variant — so compiling a
        config emits the published filename directly, no rename step."""

        for variant, (onnx, basename) in VARIANTS.items():
            with self.subTest(variant=variant):
                text = _yaml_text(f"EdgeNeXt_{variant}_config.yaml")
                self.assertIn(f"onnx_model: '{onnx}'", text)
                m = re.search(r"output_model_file_prefix: '([^']+)'", text)
                self.assertIsNotNone(m)
                self.assertEqual(f"{m.group(1)}.bin", basename)
                self.assertIn(variant, m.group(1))
                self.assertIn(f"working_dir: '{m.group(1)}'", text)

    def test_published_binding_matches_the_recipe_set(self):
        """The four recipe variants are exactly the four published assets."""

        from samples.vision.edgenext.runtime.python.model_binding import (
            BINDING_TABLE,
            list_available_assets,
        )

        self.assertEqual(
            set(BINDING_TABLE.filename_variants.values()), set(VARIANTS)
        )
        self.assertEqual(len(list_available_assets("auto")), 4)

    def test_int16_placement_counts_are_pinned(self):
        """EdgeNeXt places its cross-covariance-attention (xca) Softmax
        nodes on the BPU with int16 I/O — 3 per config (stages.1/2/3) —
        and the xx-small model adds 13 more placements (16 total)."""

        for variant, expected in INT16_PLACEMENTS.items():
            with self.subTest(variant=variant):
                text = _yaml_text(f"EdgeNeXt_{variant}_config.yaml")
                self.assertEqual(text.count("'InputType': 'int16'"), expected)
                self.assertEqual(
                    len(re.findall(r'"/stages/[^"]*xca/Softmax": \{', text)), 3
                )
                self.assertNotIn("debug_mode", text)
                self.assertNotIn("set_all_nodes_int16", text)

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
