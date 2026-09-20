# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host tests for the shared runtime metadata reader (Phase 1.5 H3).

All runtimes and models below are synthetic fixtures; none of them is board
evidence.
"""
from __future__ import annotations

import unittest


class _EnumLike:
    def __init__(self, name: str) -> None:
        self.name = name


class _QuantInfo:
    def __init__(self, *, scale, zero_point, axis=0, quant_type="SCALE") -> None:
        self.quant_type = _EnumLike(quant_type)
        self.scale = scale
        self.zero_point = zero_point
        self.axis = axis


class _MultiModelRuntime:
    model_names = ["det", "rec"]

    input_names = {"det": ["x"], "rec": ["y", "uv"]}
    input_shapes = {"det": {"x": (1, 3, 8, 8)}, "rec": {"y": (1, 8, 8, 1), "uv": (1, 4, 4, 2)}}
    input_dtypes = {"det": {"x": "NV12"}, "rec": {"y": "U8", "uv": "U8"}}
    output_names = {"det": ["out_det"], "rec": ["out_rec", "aux"]}
    output_shapes = {"det": {"out_det": (1, 2)}, "rec": {"out_rec": (1, 3), "aux": (1,)}}
    output_dtypes = {"det": {"out_det": "F32"}, "rec": {"out_rec": "int8", "aux": "F32"}}
    output_quants = {"det": {}, "rec": {"out_rec": _QuantInfo(scale=0.5, zero_point=3)}}


class RuntimeMetaTests(unittest.TestCase):
    def test_multi_model_runtime_requires_explicit_selection(self):
        from samples._shared.runtime_meta import (
            MetadataMismatchError,
            RuntimeMetadata,
        )

        with self.assertRaises(MetadataMismatchError) as ctx:
            RuntimeMetadata.from_runtime(_MultiModelRuntime())
        self.assertIn("several models", str(ctx.exception))
        self.assertIn("'det'", str(ctx.exception))

    def test_selected_model_slices_nested_attributes(self):
        from samples._shared.runtime_meta import RuntimeMetadata

        meta = RuntimeMetadata.from_runtime(_MultiModelRuntime(), "rec")
        self.assertEqual(meta.model_names, ("det", "rec"))
        self.assertEqual(meta.model_name, "rec")
        self.assertEqual(meta.input_names, ("y", "uv"))
        self.assertEqual(meta.output_names, ("out_rec", "aux"))
        self.assertEqual(meta.input_dtypes, {"y": "uint8", "uv": "uint8"})
        self.assertEqual(meta.output_dtypes["out_rec"], "int8")
        # The quant descriptor is preserved verbatim, keyed by output name.
        self.assertIn("out_rec", meta.output_quants)
        self.assertNotIn("aux", meta.output_quants)

    def test_unknown_model_selection_is_rejected(self):
        from samples._shared.runtime_meta import (
            MetadataMismatchError,
            RuntimeMetadata,
        )

        with self.assertRaises(MetadataMismatchError):
            RuntimeMetadata.from_runtime(_MultiModelRuntime(), "missing")

    def test_single_model_runtime_keeps_implicit_selection(self):
        from samples._shared.runtime_meta import RuntimeMetadata

        class _Single:
            model_names = ["only"]
            input_names = {"only": ["x"]}
            output_names = {"only": ["out"]}
            output_shapes = {"only": {"out": (1, 4)}}
            output_dtypes = {"only": {"out": "F32"}}

        meta = RuntimeMetadata.from_runtime(_Single())
        self.assertEqual(meta.model_name, "only")
        self.assertEqual(meta.output_names, ("out",))

    def test_mapping_selection_must_match_reported_models(self):
        from samples._shared.runtime_meta import (
            MetadataMismatchError,
            RuntimeMetadata,
        )

        with self.assertRaises(MetadataMismatchError):
            RuntimeMetadata.from_mapping(
                {"model_names": ["a", "b"], "model_name": "c"}
            )

    def test_flat_quant_mapping_is_keyed_by_output_name(self):
        from samples._shared.runtime_meta import RuntimeMetadata

        descriptor = _QuantInfo(scale=0.25, zero_point=0)
        meta = RuntimeMetadata.from_mapping(
            {
                "model_name": "m",
                "output_names": ["out"],
                "output_quants": {"out": descriptor},
            }
        )
        self.assertIs(meta.output_quants["out"], descriptor)

    def test_dtype_tokens_are_canonicalised(self):
        from samples._shared.runtime_meta import canonicalise_dtype

        self.assertEqual(canonicalise_dtype("F32"), "float32")
        self.assertEqual(canonicalise_dtype("hbDNNDataType.U8"), "uint8")
        self.assertEqual(canonicalise_dtype("int8"), "int8")
        self.assertEqual(canonicalise_dtype(None), None)
        self.assertEqual(canonicalise_dtype("weird"), "weird")


if __name__ == "__main__":
    unittest.main()
