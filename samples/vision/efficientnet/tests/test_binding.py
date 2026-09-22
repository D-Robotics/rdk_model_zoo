"""Contract-table and profile-coherence tests for the EfficientNet binding."""

from __future__ import annotations

from pathlib import PurePosixPath
import unittest


class BindingTableTests(unittest.TestCase):
    def _table(self):
        from samples.vision.efficientnet.runtime.python.model_binding import BINDING_TABLE

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
        variants = set(table.filename_variants.values())
        if isinstance(table.default_variant, str):
            self.assertIn(table.default_variant, variants)
        else:
            for target, variant in table.default_variant.items():
                self.assertIn(target, table.s_filename_targets + ("x5",))
                self.assertIn(variant, variants)

    def test_default_variant_follows_the_resolved_target(self):
        """B2-R1 regression: omitted variant must keep each source entrypoint's
        default per target — X5 main.py defaulted to B2 and the S wrapper to
        the per-SoC lite0 model.  A global b2 default made every S board fail
        with "no published asset" when the variant was omitted."""

        from samples.vision.efficientnet.runtime.python.model_binding import (
            BindingError,
            resolve_selection,
        )

        expected = {
            "x5": (
                "x5:efficientnet:EfficientNet_B2_224x224_nv12.bin",
                "b2",
                (224, 224),
            ),
            "s100": (
                "s:efficientnet:s100/efficientnet_lite0_224x224_nv12.hbm",
                "lite0",
                (224, 224),
            ),
            "s600": (
                "s:efficientnet:s600/efficientnet_lite0_224x224_nv12.hbm",
                "lite0",
                (224, 224),
            ),
        }
        for target, (asset_id, variant, (width, height)) in expected.items():
            with self.subTest(target=target):
                selection = resolve_selection(target)
                self.assertEqual(selection.asset_id, asset_id)
                self.assertEqual(selection.variant, variant)
                self.assertEqual(
                    (selection.contract.input_width, selection.contract.input_height),
                    (width, height),
                )

        # Explicit variants keep exact matching and per-variant geometry.
        lite2 = resolve_selection("s100", variant="lite2")
        self.assertEqual(
            lite2.asset_id,
            "s:efficientnet:s100/efficientnet_lite2_260x260_nv12.hbm",
        )
        self.assertEqual(
            (lite2.contract.input_width, lite2.contract.input_height), (260, 260)
        )

        # A b2 variant on an S target is an explicit error, never a silent
        # cross-platform substitution.
        with self.assertRaises(BindingError) as ctx:
            resolve_selection("s100", variant="b2")
        self.assertIn("No published sample asset matches", str(ctx.exception))

        # S100P rejection stays explicit with the standard message.
        with self.assertRaises(BindingError) as ctx:
            resolve_selection("s100p")
        self.assertIn("No published sample asset matches target='s100p'", str(ctx.exception))

    def test_published_listing_matches_table_and_profiles(self):
        from samples.vision.efficientnet.runtime.python.model_binding import (
            PLATFORMS,
            list_available_assets,
        )

        records = list_available_assets("auto")
        self.assertEqual(len(records), 13)
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

    def test_s100p_publishes_nothing_and_rejects_selection(self):
        from samples.vision.efficientnet.runtime.python.model_binding import (
            BindingError,
            list_available_assets,
            resolve_selection,
        )

        self.assertEqual(list_available_assets("s100p"), ())
        with self.assertRaises(BindingError):
            resolve_selection("s100p")

    def test_per_variant_geometry_matches_source_facts(self):
        """The lite series ships per-variant geometry (224/240/260/300/380);
        treating every S artifact as 224 was the exact mistake B1-R4 fixed
        for mobilenetv4 medium, so each combination is pinned here."""

        from samples.vision.efficientnet.runtime.python.model_binding import resolve_selection

        expected = {
            ('b2', 'x5'): (224, 224, 'softmax', 'source_declared_logits', 1, 'linear'),
            ('b3', 'x5'): (224, 224, 'softmax', 'source_declared_logits', 1, 'linear'),
            ('b4', 'x5'): (224, 224, 'softmax', 'source_declared_logits', 1, 'linear'),
            ('lite0', 's100'): (224, 224, 'softmax', 'source_declared_logits', 1, 'nearest'),
            ('lite0', 's600'): (224, 224, 'softmax', 'source_declared_logits', 1, 'nearest'),
            ('lite1', 's100'): (240, 240, 'softmax', 'source_declared_logits', 1, 'nearest'),
            ('lite1', 's600'): (240, 240, 'softmax', 'source_declared_logits', 1, 'nearest'),
            ('lite2', 's100'): (260, 260, 'softmax', 'source_declared_logits', 1, 'nearest'),
            ('lite2', 's600'): (260, 260, 'softmax', 'source_declared_logits', 1, 'nearest'),
            ('lite3', 's100'): (300, 300, 'softmax', 'source_declared_logits', 1, 'nearest'),
            ('lite3', 's600'): (300, 300, 'softmax', 'source_declared_logits', 1, 'nearest'),
            ('lite4', 's100'): (380, 380, 'softmax', 'source_declared_logits', 1, 'nearest'),
            ('lite4', 's600'): (380, 380, 'softmax', 'source_declared_logits', 1, 'nearest'),
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
            expected_protocol = "packed_nv12" if target == "x5" else "split_nv12"
            self.assertEqual(contract.input_protocol, expected_protocol)

    def test_bind_model_accepts_source_metadata_shapes(self):
        from samples.vision.efficientnet.runtime.python.model_binding import (
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        sizes = {
            'b2': 224, 'b3': 224, 'b4': 224,
            'lite0': 224, 'lite1': 240, 'lite2': 260,
            'lite3': 300, 'lite4': 380,
        }
        for (variant, target) in self._table().facts:
            protocol = "x5" if target == "x5" else "s"
            selection = resolve_selection(target, variant=variant)
            binding = bind_model(
                selection, runtime_metadata(protocol, sizes[variant])
            )
            if target == "x5":
                self.assertEqual(len(binding.input_names), 1)
            else:
                self.assertEqual(len(binding.input_names), 2)

    def test_bind_model_rejects_wrong_geometry(self):
        from samples.vision.efficientnet.runtime.python.model_binding import (
            MetadataMismatchError,
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        # lite1 declares 240x240; feeding the 224-shaped metadata of a lite0
        # artifact must be rejected rather than resized in secret.
        selection = resolve_selection("s100", variant="lite1")
        with self.assertRaises(MetadataMismatchError):
            bind_model(selection, runtime_metadata("s", 224))
        selection = resolve_selection("x5", variant="b2")
        with self.assertRaises(MetadataMismatchError):
            bind_model(selection, runtime_metadata("x5", wrong_geometry=True))

    def test_bind_model_keeps_vestigial_quant_descriptor_on_f32_output(self):
        # Board evidence shape (X5 smoke, 2026-09-21): published artifacts
        # ship F32 outputs that still carry a compiler quant descriptor.  The
        # raw_f32 contract gates on dtype, snapshots the descriptor, and
        # never applies it - legacy consumers ignored it too.
        from samples.vision.efficientnet.runtime.python.model_binding import (
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        sizes = {
            'b2': 224, 'b3': 224, 'b4': 224,
            'lite0': 224, 'lite1': 240, 'lite2': 260,
            'lite3': 300, 'lite4': 380,
        }
        for (variant, target) in self._table().facts:
            protocol = "x5" if target == "x5" else "s"
            selection = resolve_selection(target, variant=variant)
            binding = bind_model(
                selection,
                runtime_metadata(protocol, sizes[variant], quant_descriptor=True),
            )
            self.assertEqual(binding.output_dtype, "float32")
            self.assertIn(binding.output_name, binding.output_quants)

    def test_bind_model_rejects_quantized_output_dtype_for_raw_f32(self):
        from samples.vision.efficientnet.runtime.python.model_binding import (
            MetadataMismatchError,
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        selection = resolve_selection("x5", variant="b2")
        with self.assertRaises(MetadataMismatchError) as ctx:
            bind_model(
                selection,
                runtime_metadata("x5", output_dtype="I8", quant_descriptor=True),
            )
        self.assertIn("accepts only F32", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
