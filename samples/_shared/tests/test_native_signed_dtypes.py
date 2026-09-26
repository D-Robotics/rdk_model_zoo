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

"""Signed-dtype canonicalisation tests driven by real board metadata shapes.

The S100 board smoke (``.coordination/b7-s100-python-smoke.json``) failed with
``Unsupported native output dtype 's32'`` because the real
``HB_HBMRuntime.output_dtypes`` carries ``hbDNNDataType.S32`` enum members
(``<hbDNNDataType.S32: 8>``, see ``b7-s100-native-metadata.json``).  These
tests replay that exact metadata shape against ``canonicalise_dtype`` and the
``RuntimeMetadata.from_runtime`` path; the runtime below is a synthetic
fixture, not board evidence itself.
"""
from __future__ import annotations

import enum
import unittest


class _HbDNNDataType(enum.IntEnum):
    """Stand-in shaped like the board-only ``hbDNNDataType`` enum.

    Member values are taken from the S100 native metadata evidence
    (``U8 -> 3``, ``S32 -> 8``); the enum itself is never imported from an SDK.
    """

    U8 = 3
    S32 = 8


class _NamedOnly:
    """Object exposing just a ``name`` attribute, like a C enum wrapper."""

    def __init__(self, name: str) -> None:
        self.name = name


# The consumer-side allowed set for S-target detection heads, as enforced by
# samples/vision/yolov5/runtime/python/model_binding.py:bind_model.
_S_ALLOWED = ("float32", "int8", "uint8", "int16", "int32")


class SignedAliasTests(unittest.TestCase):
    def test_enum_members_canonicalise_to_signed_ints(self):
        from samples._shared.runtime_meta import canonicalise_dtype

        self.assertEqual(canonicalise_dtype(_HbDNNDataType.S32), "int32")
        self.assertNotEqual(canonicalise_dtype(_HbDNNDataType.S32), "float32")

    def test_named_only_objects_follow_the_name_attribute(self):
        from samples._shared.runtime_meta import canonicalise_dtype

        self.assertEqual(canonicalise_dtype(_NamedOnly("S32")), "int32")
        self.assertEqual(canonicalise_dtype(_NamedOnly("S16")), "int16")
        self.assertEqual(canonicalise_dtype(_NamedOnly("S8")), "int8")

    def test_bare_and_enum_prefixed_strings_canonicalise(self):
        from samples._shared.runtime_meta import canonicalise_dtype

        self.assertEqual(canonicalise_dtype("s32"), "int32")
        self.assertEqual(canonicalise_dtype("S32"), "int32")
        self.assertEqual(canonicalise_dtype("s16"), "int16")
        self.assertEqual(canonicalise_dtype("s8"), "int8")
        self.assertEqual(canonicalise_dtype("hbDNNDataType.S32"), "int32")
        self.assertEqual(canonicalise_dtype("hbDNNDataType.S16"), "int16")
        self.assertEqual(canonicalise_dtype("hbDNNDataType.S8"), "int8")

    def test_signed_aliases_do_not_collide_with_unsigned(self):
        from samples._shared.runtime_meta import canonicalise_dtype

        self.assertEqual(canonicalise_dtype(_HbDNNDataType.U8), "uint8")
        self.assertNotEqual(canonicalise_dtype("s8"), "uint8")
        self.assertNotEqual(canonicalise_dtype("s32"), "uint32")

    def test_existing_tokens_keep_their_canonical_forms(self):
        from samples._shared.runtime_meta import canonicalise_dtype

        self.assertEqual(canonicalise_dtype("F32"), "float32")
        self.assertEqual(canonicalise_dtype("hbDNNDataType.U8"), "uint8")
        self.assertEqual(canonicalise_dtype("i32"), "int32")
        self.assertEqual(canonicalise_dtype("int16"), "int16")
        self.assertEqual(canonicalise_dtype("f16"), "float16")
        self.assertEqual(canonicalise_dtype("nv12"), "nv12")
        self.assertEqual(canonicalise_dtype(None), None)

    def test_lanenet_signed64_public_binary_output_tokens(self):
        from samples._shared.runtime_meta import canonicalise_dtype
        for value in ("s64", "i64", "int64", "hbDNNDataType.S64", _NamedOnly("S64")):
            self.assertEqual(canonicalise_dtype(value), "int64")
        # Canonical spelling is not permission for classification bindings to
        # accept another dtype; their existing allowed sets still govern.
        self.assertNotIn("int64", _S_ALLOWED)
        self.assertEqual(canonicalise_dtype("u64"), "u64")

    def test_unknown_tokens_return_verbatim_without_guessing(self):
        from samples._shared.runtime_meta import canonicalise_dtype

        self.assertEqual(canonicalise_dtype("weird"), "weird")
        self.assertEqual(canonicalise_dtype("hbDNNDataType.WEIRD"), "hbdnndatatype.weird")
        self.assertEqual(canonicalise_dtype(_NamedOnly("S128")), "s128")


class _S100EvidenceRuntime:
    """Single-model runtime replaying the S100 YOLOv5 evidence metadata."""

    model_names = ["yolov5x_672x672_nv12"]

    input_names = {"yolov5x_672x672_nv12": ["data_uv", "data_y"]}
    input_shapes = {
        "yolov5x_672x672_nv12": {
            "data_uv": (1, 336, 336, 2),
            "data_y": (1, 672, 672, 1),
        }
    }
    input_dtypes = {
        "yolov5x_672x672_nv12": {
            "data_uv": _HbDNNDataType.U8,
            "data_y": _HbDNNDataType.U8,
        }
    }
    output_names = {"yolov5x_672x672_nv12": ["output", "1310", "1312"]}
    output_shapes = {
        "yolov5x_672x672_nv12": {
            "output": (1, 84, 84, 255),
            "1310": (1, 42, 42, 255),
            "1312": (1, 21, 21, 255),
        }
    }
    output_dtypes = {
        "yolov5x_672x672_nv12": {
            "output": _HbDNNDataType.S32,
            "1310": _HbDNNDataType.S32,
            "1312": _HbDNNDataType.S32,
        }
    }


class FromRuntimeSignedPathTests(unittest.TestCase):
    def test_evidence_runtime_yields_signed_int_outputs(self):
        from samples._shared.runtime_meta import RuntimeMetadata

        meta = RuntimeMetadata.from_runtime(_S100EvidenceRuntime())
        self.assertEqual(meta.model_name, "yolov5x_672x672_nv12")
        self.assertEqual(
            meta.input_dtypes, {"data_uv": "uint8", "data_y": "uint8"}
        )
        self.assertEqual(
            meta.output_dtypes,
            {"output": "int32", "1310": "int32", "1312": "int32"},
        )

    def test_evidence_shapes_survive_the_runtime_path(self):
        from samples._shared.runtime_meta import RuntimeMetadata

        meta = RuntimeMetadata.from_runtime(_S100EvidenceRuntime())
        self.assertEqual(
            meta.input_shapes,
            {"data_uv": (1, 336, 336, 2), "data_y": (1, 672, 672, 1)},
        )
        self.assertEqual(
            meta.output_shapes,
            {
                "output": (1, 84, 84, 255),
                "1310": (1, 42, 42, 255),
                "1312": (1, 21, 21, 255),
            },
        )

    def test_canonical_outputs_satisfy_the_s_binding_allowed_set(self):
        from samples._shared.runtime_meta import RuntimeMetadata

        meta = RuntimeMetadata.from_runtime(_S100EvidenceRuntime())
        for name in meta.output_names:
            self.assertIn(meta.output_dtypes[name], _S_ALLOWED)

    def test_enum_prefixed_string_mapping_path_canonicalises(self):
        from samples._shared.runtime_meta import RuntimeMetadata

        meta = RuntimeMetadata.from_mapping(
            {
                "model_name": "yolov5x_672x672_nv12",
                "output_names": ["output"],
                "output_shapes": {"output": (1, 84, 84, 255)},
                "output_dtypes": {"output": "hbDNNDataType.S32"},
            }
        )
        self.assertEqual(meta.output_dtypes, {"output": "int32"})


if __name__ == "__main__":
    unittest.main()
