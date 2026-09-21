"""Provenance and cross-consistency tests for the FasterNet conversion set.

The conversion directory is a verbatim rdk_x5 @ac11571 delivery: four
reference PTQ configs (S/T0/T1/T2, march ``bayes-e``, calibration
``default``).  Two source quirks are pinned rather than repaired: all four
configs share the **variant-less** output prefix ``FasterNet_224x224_nv12``
(reproducing a published basename needs a rename step), and the
``working_dir`` values are asymmetric (S uses ``model_output``, T0 appends
``_mix``, T1/T2 use the plain prefix).  No exporter script and no
``calibration_data_rgb_f32`` producer ship with the delivery.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import unittest

CONVERSION = Path(__file__).resolve().parents[1] / "conversion"

VERBATIM_SHA256 = {
    # rdk_x5 @ac11571
    "FasterNet_S_config.yaml": "f0455d5ec5b1c2b4d63f5c153b14060a1b3c17d9b02f3fbfab848239a040e867",
    "FasterNet_T0_config.yaml": "c62dd1daedf245e826dcea215ac7adec4dddc5b654b3092c6c44be67d93b371a",
    "FasterNet_T1_config.yaml": "e4a123c23edeb38e6835215ea814a1997082bcc0889ea96dfd93fe8b703a0f7a",
    "FasterNet_T2_config.yaml": "ad65f79a6e74191d17da60b727416f9c824e8597cfa4fc4d0112597aae1dd944",
}

#: variant -> (onnx input, working_dir, manifest basename)
VARIANTS = {
    "S": ("./fasternet_s.onnx", "model_output", "FasterNet_S_224x224_nv12.bin"),
    "T0": ("./fasternet_t0.onnx", "FasterNet_224x224_nv12_mix", "FasterNet_T0_224x224_nv12.bin"),
    "T1": ("./fasternet_t1.onnx", "FasterNet_224x224_nv12", "FasterNet_T1_224x224_nv12.bin"),
    "T2": ("./fasternet_t2.onnx", "FasterNet_224x224_nv12", "FasterNet_T2_224x224_nv12.bin"),
}

#: int16 node placements per config, counted from the pinned source bytes
#: (only T0 places nodes — 2 partial-conv related placements).
INT16_PLACEMENTS = {"S": 0, "T0": 2, "T1": 0, "T2": 0}


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
                self.assertIn("calibration_type: 'default'", text)

    def test_output_prefix_is_variant_less_and_needs_a_rename(self):
        """All four configs emit ``FasterNet_224x224_nv12`` — prefix + '.bin'
        is NOT any manifest basename, so reproducing a published artifact
        requires an explicit rename.  The ONNX input names do carry the
        variant (lowercase); only the output side is variant-less."""

        for variant, (onnx, _, basename) in VARIANTS.items():
            with self.subTest(variant=variant):
                text = _yaml_text(f"FasterNet_{variant}_config.yaml")
                self.assertIn(f"onnx_model: '{onnx}'", text)
                m = re.search(r"output_model_file_prefix: '([^']+)'", text)
                self.assertIsNotNone(m)
                prefix = m.group(1)
                self.assertNotIn(variant, prefix, "the source prefix is variant-less; do not fake agreement")
                self.assertNotEqual(f"{prefix}.bin", basename)
                self.assertTrue(basename.startswith(f"FasterNet_{variant}_"))

    def test_working_dir_asymmetry_is_pinned(self):
        """The source configs disagree on working_dir (S: ``model_output``,
        T0: prefix + ``_mix``, T1/T2: plain prefix) — preserved verbatim so
        reproduction notes stay honest."""

        for variant, (_, working_dir, _) in VARIANTS.items():
            with self.subTest(variant=variant):
                text = _yaml_text(f"FasterNet_{variant}_config.yaml")
                self.assertIn(f"working_dir: '{working_dir}'", text)

    def test_published_binding_matches_the_recipe_set(self):
        from samples.vision.fasternet.runtime.python.model_binding import (
            BINDING_TABLE,
            list_available_assets,
        )

        # recipe keys keep the source's capital S/T spelling; the binding
        # table's variant ids are lowercase (unified-family convention)
        self.assertEqual(
            set(BINDING_TABLE.filename_variants.values()),
            {v.lower() for v in VARIANTS},
        )
        self.assertEqual(len(list_available_assets("auto")), 4)

    def test_int16_placement_counts_are_pinned(self):
        for variant, expected in INT16_PLACEMENTS.items():
            with self.subTest(variant=variant):
                text = _yaml_text(f"FasterNet_{variant}_config.yaml")
                self.assertEqual(text.count("'InputType': 'int16'"), expected)
                self.assertNotRegex(text, r'"/stages/[^"]*Softmax')
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
