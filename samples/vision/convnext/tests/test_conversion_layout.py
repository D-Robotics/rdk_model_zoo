"""Provenance and cross-consistency tests for the ConvNeXt conversion set.

The conversion directory is a verbatim rdk_x5 @ac11571 delivery: three PTQ
reference configs (atto/femto/nano, march ``bayes-e``, calibration
``default``) while the X5 manifest publishes exactly one asset (atto).
Adjusting a pinned file is a source-branch change, not a local edit.  The
source recipes carry quirks these tests pin instead of silently repairing:
the ONNX references are mutually inconsistent (atto points at
``convnext_femto.onnx``, femto at an external common-zoo ``convnext_atto.onnx``
path, nano at ``convnext_pico.onnx``), and the shared output prefix is
variant-less (``ConvNeXt-deploy_224x224_nv12``), so reproducing the published
atto artifact needs an explicit rename step.  No exporter script and no
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
    "ConvNeXt_atto.yaml": "3a464fdc0f7ac0e757564287a2f58ba45413bee3adddbe571e3a7937915fa8aa",
    "ConvNeXt_femto.yaml": "6f220eb14d1251bad0383b462254d5b9880aa1c1fe899b2b63cafdd2c27c1b0e",
    "ConvNeXt_nano.yaml": "f844e099d6080417fa44d76cace538574d6322671e4d616c6d0c91e85116437f",
}

#: The only filename the X5 manifest publishes for this sample.
MANIFEST_BASENAME = "ConvNeXt_atto_224x224_nv12.bin"
#: The femto recipe points outside the sample tree (common model zoo).
EXTERNAL_ONNX_REFERENCE = (
    "../../../01_common/model_zoo/mapper/classification/ConvNeXt/convnext_atto.onnx"
)
#: int16 node placements per config, counted from the pinned source bytes.
INT16_PLACEMENTS = {"ConvNeXt_atto.yaml": 8, "ConvNeXt_femto.yaml": 5, "ConvNeXt_nano.yaml": 8}


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
                # Same numeric family as the other X5 classification configs.
                self.assertIn("mean_value: 123.675 116.28 103.53", text)
                self.assertIn("scale_value: 0.01712475 0.017507 0.01742919", text)
                self.assertIn("cal_data_dir: './calibration_data_rgb_f32'", text)
                self.assertIn("cal_data_type: 'float32'", text)
                self.assertIn("calibration_type: 'default'", text)

    def test_onnx_references_are_mutually_inconsistent_as_shipped(self):
        """Pin the source quirks verbatim: the atto config consumes
        ``convnext_femto.onnx``, the femto config points at an external
        common-zoo ``convnext_atto.onnx`` path, and the nano config points
        at ``convnext_pico.onnx``.  These are disclosed facts of the source
        delivery, not repaired here; no coherent exporter flow is claimed."""

        self.assertIn("onnx_model: './convnext_femto.onnx'", _yaml_text("ConvNeXt_atto.yaml"))
        self.assertIn(
            f"onnx_model: '{EXTERNAL_ONNX_REFERENCE}'", _yaml_text("ConvNeXt_femto.yaml")
        )
        self.assertIn("onnx_model: './convnext_pico.onnx'", _yaml_text("ConvNeXt_nano.yaml"))

    def test_output_prefix_is_variant_less_and_needs_a_rename(self):
        """All three configs share the prefix ``ConvNeXt-deploy_224x224_nv12``:
        prefix + '.bin' is NOT the manifest basename, so reproducing the
        published atto artifact requires an explicit rename step.  Pin the
        mismatch instead of hiding it."""

        for name in VERBATIM_SHA256:
            with self.subTest(file=name):
                text = _yaml_text(name)
                m = re.search(r"output_model_file_prefix: '([^']+)'", text)
                self.assertIsNotNone(m)
                prefix = m.group(1)
                self.assertNotIn("atto", prefix, "the source prefix is variant-less; do not fake agreement")
                self.assertNotEqual(f"{prefix}.bin", MANIFEST_BASENAME)
        self.assertTrue(MANIFEST_BASENAME.startswith("ConvNeXt_atto_"))

    def test_only_atto_is_published_femto_and_nano_are_recipes_only(self):
        """The manifest carries exactly one ConvNeXt row (atto).  femto and
        nano exist as conversion recipes with no published asset — the
        runtime binding must not grow variants for them."""

        from samples.vision.convnext.runtime.python.model_binding import (
            BINDING_TABLE,
            list_available_assets,
        )

        records = list_available_assets("auto")
        self.assertEqual([r.filename for r in records], [MANIFEST_BASENAME])
        self.assertEqual(set(BINDING_TABLE.filename_variants.values()), {"atto"})
        self.assertNotIn("femto", BINDING_TABLE.filename_variants.values())
        self.assertNotIn("nano", BINDING_TABLE.filename_variants.values())

    def test_int16_placement_counts_are_pinned(self):
        """The node_info tables place 8/5/8 int16 nodes (atto/femto/nano) —
        pin the counts so a dropped/added placement is visible.  No Softmax
        placement exists in this family and no debug dump or forced
        all-int16 optimization ships."""

        for name, expected in INT16_PLACEMENTS.items():
            with self.subTest(file=name):
                text = _yaml_text(name)
                self.assertEqual(text.count("'InputType': 'int16'"), expected)
                self.assertEqual(text.count("node_info"), 1)
                # ConvNeXt places depthwise/normalization nodes, not Softmax.
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
