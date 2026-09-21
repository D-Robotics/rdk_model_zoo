"""Contract-table and profile-coherence tests for the FastViT binding."""

from __future__ import annotations

from pathlib import PurePosixPath
import unittest


class BindingTableTests(unittest.TestCase):
    def _table(self):
        from samples.vision.fastvit.runtime.python.model_binding import BINDING_TABLE

        return BINDING_TABLE

    def test_every_published_filename_has_contract_facts(self):
        table = self._table()
        for filename, variant in table.filename_variants.items():
            target = "x5" if filename.endswith(".bin") else filename.split("/")[0]
            self.assertIn(
                (variant, target),
                table.facts,
                f"filename {filename!r} maps to no facts entry",
            )
        self.assertIn(table.default_variant, ('s12', 'sa12', 't12', 't8'))

    def test_published_listing_matches_table_and_profiles(self):
        from samples.vision.fastvit.runtime.python.model_binding import (
            PLATFORMS,
            list_available_assets,
        )

        records = list_available_assets("auto")
        self.assertEqual(len(records), 4)
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
            self.assertTrue(
                record.source_manifest.replace("\\", "/").endswith("x5/models.yaml"),
                f"unexpected manifest source: {record.source_manifest}",
            )
        self.assertEqual(
            sorted(record.filename for record in records),
            [
                "FastViT_S12_224x224_nv12.bin",
                "FastViT_SA12_224x224_nv12.bin",
                "FastViT_T12_224x224_nv12.bin",
                "FastViT_T8_224x224_nv12.bin",
            ],
        )

    def test_s_targets_publish_nothing_and_reject_selection(self):
        from samples.vision.fastvit.runtime.python.model_binding import (
            BindingError,
            list_available_assets,
            resolve_selection,
        )

        for target in ("s100", "s100p", "s600"):
            with self.subTest(target=target):
                self.assertEqual(list_available_assets(target), ())
                with self.assertRaises(BindingError):
                    resolve_selection(target)

    def test_default_variant_preserves_source_entrypoint_default(self):
        # The source main.py defaulted to the S12 (lowercase id `s12`) artifact; the unified table
        # keeps that default so zero-argument selection is unchanged.
        table = self._table()
        self.assertEqual(table.default_variant, "s12")

    def test_per_variant_geometry_matches_source_facts(self):
        from samples.vision.fastvit.runtime.python.model_binding import resolve_selection

        expected = {
            ('s12', 'x5'): (224, 224, 'softmax', 'source_declared_logits', 1, 'linear'),
            ('sa12', 'x5'): (224, 224, 'softmax', 'source_declared_logits', 1, 'linear'),
            ('t12', 'x5'): (224, 224, 'softmax', 'source_declared_logits', 1, 'linear'),
            ('t8', 'x5'): (224, 224, 'softmax', 'source_declared_logits', 1, 'linear'),
        }
        self.assertEqual(set(expected), set(self._table().facts))
        for (variant, target), (height, width, policy, semantics, resize, interp) in expected.items():
            selection = resolve_selection(target, variant=variant)
            contract = selection.contract
            self.assertEqual(
                (contract.input_height, contract.input_width), (height, width),
                f"{variant}/{target} geometry",
            )
            self.assertEqual(contract.output_score_policy, policy, f"{variant}/{target} policy")
            self.assertEqual(contract.output_semantics, semantics, f"{variant}/{target} semantics")
            self.assertEqual(contract.resize_type, resize, f"{variant}/{target} resize")
            self.assertEqual(
                contract.resize_interpolation, interp, f"{variant}/{target} interp"
            )
            self.assertEqual(contract.class_count, 1000)
            self.assertEqual(contract.output_transform, "raw_f32")
            self.assertEqual(contract.input_protocol, "packed_nv12")

    def test_bind_model_accepts_source_metadata_shapes(self):
        from samples.vision.fastvit.runtime.python.model_binding import (
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        for (variant, target) in self._table().facts:
            selection = resolve_selection(target, variant=variant)
            binding = bind_model(selection, runtime_metadata())
            self.assertEqual(len(binding.input_names), 1)

    def test_bind_model_rejects_wrong_geometry(self):
        from samples.vision.fastvit.runtime.python.model_binding import (
            MetadataMismatchError,
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        selection = resolve_selection("x5", variant="t12")
        with self.assertRaises(MetadataMismatchError):
            bind_model(selection, runtime_metadata(wrong_geometry=True))

    def test_bind_model_keeps_vestigial_quant_descriptor_on_f32_output(self):
        # Board evidence shape (X5 smoke, 2026-09-21): published artifacts
        # ship F32 outputs that still carry a compiler quant descriptor.  The
        # raw_f32 contract gates on dtype, snapshots the descriptor, and
        # never applies it - legacy consumers ignored it too.
        from samples.vision.fastvit.runtime.python.model_binding import (
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        for (variant, target) in self._table().facts:
            selection = resolve_selection(target, variant=variant)
            binding = bind_model(selection, runtime_metadata(quant_descriptor=True))
            self.assertEqual(binding.output_dtype, "float32")
            self.assertIn(binding.output_name, binding.output_quants)

    def test_bind_model_rejects_quantized_output_dtype_for_raw_f32(self):
        from samples.vision.fastvit.runtime.python.model_binding import (
            MetadataMismatchError,
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        selection = resolve_selection("x5", variant="t12")
        with self.assertRaises(MetadataMismatchError) as ctx:
            bind_model(
                selection,
                runtime_metadata(output_dtype="I8", quant_descriptor=True),
            )
        self.assertIn("accepts only F32", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
