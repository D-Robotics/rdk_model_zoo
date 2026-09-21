"""Provenance and cross-consistency tests for the FastViT conversion set.

The conversion directory is a verbatim rdk_x5 @ac11571 delivery: four
reference PTQ configs (S12/SA12/T12/T8, march ``bayes-e``, calibration
``default``).  Two source quirks are pinned rather than repaired: every
``onnx_model`` points at an **external common model-zoo path**
(``../../../01_common/model_zoo/mapper/classification/FastViT/...``)
outside this sample tree, and all four configs share the **variant-less**
output prefix ``FastViT_224x224_nv12`` (reproducing a published basename
needs a rename step).  No exporter script and no
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
    "FastViT_S12_config.yaml": "50c5b40ab3d801ad72eae45a6927dcce4074cf48af90d62d0b974236d46eb8d2",
    "FastViT_SA12_config.yaml": "612f9e668d2a30549c33d72595bc84846d05c2404960b31276f174b8c6ddc8fe",
    "FastViT_T12_config.yaml": "17b23a8dc23423e499d68d0f9ec3cacf5b2184148fdaa710f20a125c7509a5e2",
    "FastViT_T8_config.yaml": "79ab7b5478b3978af871feb81c90e70b87838fa65d5d922cc8d14becbf7d6a0f",
}

#: variant -> (onnx basename, manifest basename); the onnx_model entries all
#: carry the same external ../../..01_common prefix (pinned below).
EXTERNAL_PREFIX = "../../../01_common/model_zoo/mapper/classification/FastViT"
VARIANTS = {
    "S12": ("fastvit_s12.onnx", "FastViT_S12_224x224_nv12.bin"),
    "SA12": ("fastvit_sa12.onnx", "FastViT_SA12_224x224_nv12.bin"),
    "T12": ("fastvit_t12.onnx", "FastViT_T12_224x224_nv12.bin"),
    "T8": ("fastvit_t8.onnx", "FastViT_T8_224x224_nv12.bin"),
}

#: int16 node placements per config, counted from the pinned source bytes.
INT16_PLACEMENTS = {"S12": 5, "SA12": 6, "T12": 4, "T8": 10}


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

    def test_onnx_inputs_point_at_the_external_common_zoo(self):
        """As shipped, every config consumes its ONNX from the shared
        ``01_common`` model-zoo path outside the sample tree — pinned
        verbatim, not repaired; no exporter flow is claimed for it."""

        for variant, (onnx, _) in VARIANTS.items():
            with self.subTest(variant=variant):
                text = _yaml_text(f"FastViT_{variant}_config.yaml")
                self.assertIn(f"onnx_model: '{EXTERNAL_PREFIX}/{onnx}'", text)

    def test_output_prefix_is_variant_less_and_needs_a_rename(self):
        """All four configs emit ``FastViT_224x224_nv12`` into the shared
        working_dir ``FastViT_224x224_nv12_mix`` — prefix + '.bin' is NOT
        any manifest basename, so reproducing a published artifact requires
        an explicit rename."""

        for variant, (_, basename) in VARIANTS.items():
            with self.subTest(variant=variant):
                text = _yaml_text(f"FastViT_{variant}_config.yaml")
                m = re.search(r"output_model_file_prefix: '([^']+)'", text)
                self.assertIsNotNone(m)
                prefix = m.group(1)
                self.assertNotIn(variant.lower(), prefix, "the source prefix is variant-less; do not fake agreement")
                self.assertNotEqual(f"{prefix}.bin", basename)
                self.assertIn("working_dir: 'FastViT_224x224_nv12_mix'", text)
                self.assertTrue(basename.startswith(f"FastViT_{variant}_"))

    def test_published_binding_matches_the_recipe_set(self):
        from samples.vision.fastvit.runtime.python.model_binding import (
            BINDING_TABLE,
            list_available_assets,
        )

        # recipe keys keep the source spelling; the binding table's variant
        # ids are lowercase (unified-family convention)
        self.assertEqual(
            set(BINDING_TABLE.filename_variants.values()),
            {v.lower() for v in VARIANTS},
        )
        self.assertEqual(len(list_available_assets("auto")), 4)

    def test_int16_placement_counts_are_pinned(self):
        for variant, expected in INT16_PLACEMENTS.items():
            with self.subTest(variant=variant):
                text = _yaml_text(f"FastViT_{variant}_config.yaml")
                self.assertEqual(text.count("'InputType': 'int16'"), expected)
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
