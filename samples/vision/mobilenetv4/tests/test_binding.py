"""Contract-table and profile-coherence tests for the MobileNetV4 binding."""

from __future__ import annotations

from pathlib import PurePosixPath
import unittest


class BindingTableTests(unittest.TestCase):
    def _table(self):
        from samples.vision.mobilenetv4.runtime.python.cli import BINDING_TABLE

        return BINDING_TABLE

    def test_every_published_filename_has_contract_facts(self):
        table = self._table()
        for filename, variant in table.filename_variants.items():
            suffix = "bin" if "/" not in filename else "hbm"
            target = "x5" if suffix == "bin" else filename.split("/")[0]
            self.assertIn(
                (variant, target),
                table.facts,
                f"filename {filename!r} maps to no facts entry",
            )
        self.assertIn(table.default_variant, ('small', 'medium'))

    def test_published_listing_matches_table_and_profiles(self):
        from samples.vision.mobilenetv4.runtime.python.cli import (
            PLATFORMS,
            list_available_assets,
        )

        records = list_available_assets("auto")
        self.assertTrue(records)
        table = self._table()
        for record in records:
            self.assertEqual(
                record.variant, table.filename_variants[record.filename]
            )
            profile = PLATFORMS[record.target]
            # H5 coherence: manifest rows and platform profiles agree on the
            # published artifact format and directory layout.
            self.assertEqual(
                record.model_format, profile.model_format.lstrip(".")
            )
            parent = PurePosixPath(record.filename).parent
            parent_str = "" if parent == PurePosixPath(".") else parent.as_posix()
            self.assertEqual(parent_str, profile.model_subdir)
            # The row was read from the platform-group manifest that owns it.
            expected_group = "x5" if record.target == "x5" else "s"
            self.assertTrue(
                record.source_manifest.replace("\\", "/").endswith(
                    f"{expected_group}/models.yaml"
                ),
                f"unexpected manifest source: {record.source_manifest}",
            )

    def test_s100p_publishes_both_variants(self):
        from samples.vision.mobilenetv4.runtime.python.cli import (
            list_available_assets,
            resolve_selection,
        )

        records = list_available_assets("s100p")
        self.assertEqual(
            {(r.variant, r.filename) for r in records},
            {
                ("small", "s100p/mobilenetv4_conv_small_nashm_224x224_nv12.hbm"),
                ("medium", "s100p/mobilenetv4_conv_medium_nashm_224x224_nv12.hbm"),
            },
        )
        for variant in ("small", "medium"):
            selection = resolve_selection("s100p", variant=variant)
            self.assertEqual(selection.target, "s100p")
            self.assertEqual(selection.contract.input_protocol, "split_nv12")
            self.assertEqual(selection.model_path.parent.name, "s100p")

    def test_every_target_publishes_both_variants_with_hash(self):
        from samples.vision.mobilenetv4.runtime.python.cli import list_available_assets
        from utils.py_utils.assets import resolve_asset

        for target in ("x5", "s100", "s100p", "s600"):
            records = list_available_assets(target)
            self.assertEqual({r.variant for r in records}, {"small", "medium"}, target)
            for record in records:
                asset = resolve_asset(record.asset_id)
                self.assertRegex(asset.sha256 or "", r"^[0-9a-f]{64}$", record.asset_id)
                self.assertTrue(
                    asset.url.startswith(
                        "https://rdk-model-zoo.oss-cn-beijing.aliyuncs.com/models/"
                        f"mobilenetv4/mobilenetv4/cls/conv-{record.variant}-224/{target}/"
                    ),
                    asset.url,
                )

    def test_per_target_contracts_match_source_facts(self):
        from samples.vision.mobilenetv4.runtime.python.cli import resolve_selection

        # Every published model is 224x224 and uses the timm evaluation geometry
        # (resize type 2): shorter edge int(224 / crop_pct), then a center crop.
        shorter = {"small": 256, "medium": 235}
        for variant in ("small", "medium"):
            for target in ("x5", "s100", "s100p", "s600"):
                contract = resolve_selection(target, variant=variant).contract
                label = f"{variant}/{target}"
                self.assertEqual((contract.input_height, contract.input_width), (224, 224), label)
                self.assertEqual(contract.output_score_policy, "softmax", label)
                self.assertEqual(contract.output_semantics, "source_declared_logits", label)
                self.assertEqual(contract.resize_type, 2, label)
                self.assertEqual(contract.resize_shorter, shorter[variant], label)
                self.assertEqual(contract.class_count, 1000)
                self.assertEqual(contract.output_transform, "raw_f32")
                expected_protocol = "packed_nv12" if target == "x5" else "split_nv12"
                self.assertEqual(contract.input_protocol, expected_protocol)

    def test_bind_model_accepts_source_metadata_shapes(self):
        from samples.vision.mobilenetv4.runtime.python.cli import (
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        for (variant, target) in self._table().facts:
            protocol = "x5" if target == "x5" else "s"
            selection = resolve_selection(target, variant=variant)
            binding = bind_model(
                selection, runtime_metadata(protocol, variant=variant)
            )
            if target == "x5":
                self.assertEqual(len(binding.input_names), 1)
            else:
                self.assertEqual(len(binding.input_names), 2)

    def test_bind_model_rejects_wrong_geometry(self):
        from samples.vision.mobilenetv4.runtime.python.cli import (
            MetadataMismatchError,
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        variant, target = ('small', 'x5')
        protocol = "x5" if target == "x5" else "s"
        selection = resolve_selection(target, variant=variant)
        with self.assertRaises(MetadataMismatchError):
            bind_model(selection, runtime_metadata(protocol, variant=variant, wrong_geometry=True))

    def test_bind_model_keeps_vestigial_quant_descriptor_on_f32_output(self):
        # Board evidence (X5 smoke, 2026-09-21): published artifacts ship F32
        # outputs that still carry a compiler quant descriptor.  The raw_f32
        # contract gates on dtype, snapshots the descriptor, and never applies
        # it - legacy consumers ignored it too.
        from samples.vision.mobilenetv4.runtime.python.cli import (
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        for (variant, target) in self._table().facts:
            protocol = "x5" if target == "x5" else "s"
            selection = resolve_selection(target, variant=variant)
            binding = bind_model(
                selection,
                runtime_metadata(protocol, variant=variant, quant_descriptor=True),
            )
            self.assertEqual(binding.output_dtype, "float32")
            self.assertIn(binding.output_name, binding.output_quants)

    def test_bind_model_rejects_quantized_output_dtype_for_raw_f32(self):
        from samples.vision.mobilenetv4.runtime.python.cli import (
            MetadataMismatchError,
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        variant, target = ('small', 'x5')
        protocol = "x5" if target == "x5" else "s"
        selection = resolve_selection(target, variant=variant)
        with self.assertRaises(MetadataMismatchError) as ctx:
            bind_model(
                selection,
                runtime_metadata(
                    protocol, variant=variant, output_dtype="I8", quant_descriptor=True
                ),
            )
        self.assertIn("accepts only F32", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
