"""Host contract tests for the ResNet pilot binding.

The fixtures model the metadata exposed by the existing ``hbm_runtime``
wrappers.  They are deliberately small and do not pretend to be board
execution evidence.
"""

from __future__ import annotations

import unittest


def _metadata(*, protocol: str = "x5", output_shape=None,
              output_dtype: str = "float32", output_semantics: str | None = None):
    from samples.vision.resnet.runtime.python.model_binding import RuntimeMetadata

    if protocol == "x5":
        inputs = {"data": (1, 3, 224, 224)}
        input_names = ["data"]
        output_name = "prob"
        if output_shape is None:
            output_shape = (1, 1000, 1, 1)
    else:
        inputs = {
            "input_y": (1, 224, 224, 1),
            "input_uv": (1, 112, 112, 2),
        }
        input_names = ["input_y", "input_uv"]
        output_name = "output"
        if output_shape is None:
            output_shape = (1, 1000)
    values = {
        "model_name": "resnet18_224x224_nv12",
        "input_names": input_names,
        "input_shapes": inputs,
        "input_dtypes": {
            name: ("NV12" if protocol == "x5" else "U8")
            for name in input_names
        },
        "output_names": [output_name],
        "output_shapes": {output_name: output_shape},
        "output_dtypes": {output_name: output_dtype},
    }
    if output_semantics is not None:
        values["output_semantics"] = output_semantics
    return RuntimeMetadata.from_mapping(values)


class BindingTests(unittest.TestCase):
    def test_x5_asset_binds_to_packed_nv12_and_unverified_scores(self):
        from samples.vision.resnet.runtime.python.model_binding import (
            bind_model,
            resolve_selection,
        )

        selection = resolve_selection("x5")
        binding = bind_model(selection, _metadata(protocol="x5"))

        self.assertEqual(binding.contract.input_protocol, "packed_nv12")
        self.assertEqual(binding.input_names, ("data",))
        self.assertEqual(binding.output_name, "prob")
        self.assertEqual(binding.contract.output_semantics, "unverified_score_vector")
        self.assertEqual(binding.contract.output_score_policy, "legacy_softmax")

    def test_x5_accepts_runtime_nv12_dtype_token(self):
        from samples.vision.resnet.runtime.python.model_binding import (
            bind_model,
            resolve_selection,
        )

        metadata = _metadata(protocol="x5")
        metadata = metadata.from_mapping(
            {
                "model_name": metadata.model_name,
                "input_names": metadata.input_names,
                "input_shapes": metadata.input_shapes,
                "input_dtypes": {"data": "hbDNNDataType.NV12"},
                "output_names": metadata.output_names,
                "output_shapes": metadata.output_shapes,
                "output_dtypes": metadata.output_dtypes,
            }
        )
        binding = bind_model(resolve_selection("x5"), metadata)
        self.assertEqual(binding.input_names, ("data",))

    def test_direct_runtime_metadata_defaults_are_safe_mappings(self):
        from samples.vision.resnet.runtime.python.model_binding import (
            RuntimeMetadata,
            bind_model,
            resolve_selection,
        )

        metadata = RuntimeMetadata(
            model_name="resnet18_224x224_nv12",
            input_names=("data",),
            input_shapes={"data": (1, 3, 224, 224)},
            output_names=("prob",),
            output_shapes={"prob": (1, 1000, 1, 1)},
            input_dtypes={"data": "nv12"},
            output_dtypes={"prob": "float32"},
        )
        self.assertEqual(bind_model(resolve_selection("x5"), metadata).output_name, "prob")

    def test_missing_runtime_dtypes_are_rejected(self):
        from samples.vision.resnet.runtime.python.model_binding import (
            MetadataMismatchError,
            RuntimeMetadata,
            bind_model,
            resolve_selection,
        )

        metadata = RuntimeMetadata.from_mapping(
            {
                "model_name": "resnet18_224x224_nv12",
                "input_names": ["data"],
                "input_shapes": {"data": (1, 3, 224, 224)},
                "output_names": ["prob"],
                "output_shapes": {"prob": (1, 1000, 1, 1)},
            }
        )
        with self.assertRaises(MetadataMismatchError):
            bind_model(resolve_selection("x5"), metadata)

    def test_s100_and_s600_assets_bind_to_split_nv12(self):
        from samples.vision.resnet.runtime.python.model_binding import (
            bind_model,
            resolve_selection,
        )

        for target in ("s100", "s600"):
            with self.subTest(target=target):
                selection = resolve_selection(target)
                binding = bind_model(selection, _metadata(protocol="s"))
                self.assertEqual(binding.contract.input_protocol, "split_nv12")
                self.assertEqual(binding.y_input_name, "input_y")
                self.assertEqual(binding.uv_input_name, "input_uv")

    def test_unpublished_target_and_unknown_asset_are_rejected(self):
        from samples.vision.resnet.runtime.python.model_binding import (
            UnsupportedAssetError,
            resolve_selection,
        )

        with self.assertRaises(UnsupportedAssetError):
            resolve_selection("s100p", asset_id="resnet18")
        with self.assertRaises(UnsupportedAssetError):
            resolve_selection("x5", asset_id="made-up-resnet")

    def test_wrong_output_shape_or_semantics_is_rejected(self):
        from samples.vision.resnet.runtime.python.model_binding import (
            MetadataMismatchError,
            bind_model,
            resolve_selection,
        )

        selection = resolve_selection("x5")
        with self.assertRaises(MetadataMismatchError):
            bind_model(selection, _metadata(output_shape=(1, 999)))
        with self.assertRaises(MetadataMismatchError):
            bind_model(selection, _metadata(output_semantics="not-a-score-vector"))

    def test_integer_output_without_declared_scale_is_rejected(self):
        from samples.vision.resnet.runtime.python.model_binding import (
            MetadataMismatchError,
            bind_model,
            resolve_selection,
        )

        selection = resolve_selection("x5")
        with self.assertRaises(MetadataMismatchError):
            bind_model(selection, _metadata(output_dtype="int8"))

    def test_explicit_custom_path_requires_a_known_contract(self):
        from samples.vision.resnet.runtime.python.model_binding import (
            UnsupportedAssetError,
            resolve_selection,
        )

        with self.assertRaises(UnsupportedAssetError):
            resolve_selection("x5", model_path="custom.bin")


if __name__ == "__main__":
    unittest.main()
