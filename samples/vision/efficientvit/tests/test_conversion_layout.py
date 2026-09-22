"""Provenance and cross-consistency tests for the EfficientViT conversion set.

The conversion directory is a verbatim rdk_x5 @ac11571 delivery: the single
``EfficientViT_MSRA_m5_config.yaml`` reference PTQ config (march ``bayes-e``,
calibration ``max``).  Adjusting a pinned file is a source-branch change, not
a local edit.  Unlike the EfficientFormerV2 sibling, this YAML's output
prefix is variant-less (``msra``, not ``m5``) and does **not** reproduce the
manifest basename; the anchors below pin that gap plus the retained
source-recipe facts (no ONNX exporter, no calibration producer, the unique
``0.99999`` percentile, and the 28-node Softmax int16 placement table).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import unittest

CONVERSION = Path(__file__).resolve().parents[1] / "conversion"

VERBATIM_SHA256 = {
    # rdk_x5 @ac11571
    "EfficientViT_MSRA_m5_config.yaml": "65915fea82515c66457d1ac48051bd8172f97155e89dc9b9a65499dfcb13312c",
}

MANIFEST_BASENAME = "EfficientViT_m5_224x224_nv12.bin"
ONNX_INPUT = "./efficientvit_m5.onnx"
WORKING_DIR = "EfficientViT_msra_224x224_nv12"


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
    def test_x5_config_declares_the_classification_numeric_family(self):
        text = _yaml_text("EfficientViT_MSRA_m5_config.yaml")
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

    def test_onnx_input_carries_the_variant_and_percentile_is_pinned(self):
        """The ONNX name does carry the m5 identity (unlike the output
        prefix below), and this delivery alone uses the 0.99999 max
        percentile — pin both so a quiet re-tune is caught."""

        text = _yaml_text("EfficientViT_MSRA_m5_config.yaml")
        self.assertIn(f"onnx_model: '{ONNX_INPUT}'", text)
        self.assertIn("max_percentile: 0.99999", text)
        self.assertIn(f"working_dir: '{WORKING_DIR}'", text)

    def test_output_prefix_is_variant_less_and_needs_a_rename(self):
        """Unlike efficientformerv2, this delivery's output prefix is
        ``EfficientViT_msra_224x224_nv12``: prefix + '.bin' is NOT the
        manifest basename, so reproducing the published artifact requires
        an explicit rename step.  Pin the mismatch instead of hiding it."""

        text = _yaml_text("EfficientViT_MSRA_m5_config.yaml")
        m = re.search(r"output_model_file_prefix: '([^']+)'", text)
        self.assertIsNotNone(m)
        prefix = m.group(1)
        self.assertNotIn("m5", prefix, "the source prefix is variant-less; do not fake agreement")
        self.assertNotEqual(f"{prefix}.bin", MANIFEST_BASENAME)
        # The manifest basename itself is the m5 artifact, so a rename from
        # the emitted name is a real, disclosed step.
        self.assertTrue(MANIFEST_BASENAME.startswith("EfficientViT_m5_"))

    def test_softmax_int16_placement_table_is_pinned(self):
        """This config places 28 attention Softmax nodes on the BPU with
        int16 I/O — the CGA (cascaded group attention) structure.  Pin the
        count so a dropped/added placement is visible."""

        text = _yaml_text("EfficientViT_MSRA_m5_config.yaml")
        placements = re.findall(r'"/stages/[^"]+Softmax[^"]*": \{', text)
        self.assertEqual(len(placements), 28)
        self.assertEqual(text.count("'InputType': 'int16'"), 28)
        # No debug dump or forced all-int16 optimization in this delivery.
        self.assertNotIn("debug_mode", text)
        self.assertNotIn("set_all_nodes_int16", text)

    def test_no_exporter_or_calibration_producer_is_claimed(self):
        """The X5 delivery ships a reference config only: no exporter script
        for the ONNX input and no producer for ``calibration_data_rgb_f32``
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
