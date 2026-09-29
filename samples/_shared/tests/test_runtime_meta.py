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

import copy
import json
import unittest
from dataclasses import asdict

import numpy as np


class _EnumLike:
    def __init__(self, name: str) -> None:
        self.name = name


class _QuantInfo:
    def __init__(self, *, scale, zero_point, axis=0, quant_type="SCALE") -> None:
        self.quant_type = _EnumLike(quant_type)
        self.scale = scale
        self.zero_point = zero_point
        self.axis = axis


class _BoardQuantParams:
    """SDK-like quant descriptor: readable, but any copy is refused.

    ``hbm_runtime.HB_HBMRuntime.QuantParams`` fails ``copy.deepcopy`` with
    ``TypeError: cannot pickle ...`` on the X5 board (evidence 2026-09-24);
    this fixture reproduces exactly that failure surface.
    """

    def __init__(self, *, quant_type="SCALE", scale=1.0, zero_point=0, axis=0, **extra):
        self.quant_type = quant_type if isinstance(quant_type, _EnumLike) else _EnumLike(quant_type)
        self.scale = scale
        self.zero_point = zero_point
        self.axis = axis
        for name, value in extra.items():
            setattr(self, name, value)

    def __deepcopy__(self, memo):
        raise TypeError("cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object")

    def __copy__(self):
        raise TypeError("cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object")


class _SingleModelRuntime:
    model_names = ["det"]
    input_names = {"det": ["x"]}
    input_shapes = {"det": {"x": (1, 3, 8, 8)}}
    input_dtypes = {"det": {"x": "NV12"}}
    output_names = {"det": ["out", "aux"]}
    output_shapes = {"det": {"out": (1, 8, 8, 255), "aux": (1, 2)}}
    output_dtypes = {"det": {"out": "int8", "aux": "F32"}}
    input_strides = {"det": {"x": (1, 1, 8, 1)}}
    output_strides = {"det": {"out": (2040, 255, 1, 1)}}
    output_quants = {
        "det": {
            "out": _BoardQuantParams(
                quant_type="SCALE",
                scale=np.array([0.25, 0.5], np.float32),
                zero_point=np.array([3, -7], np.int32),
                axis=3,
            ),
            "aux": _BoardQuantParams(quant_type="NONE", scale=np.float32(0.0), zero_point=0),
        }
    }


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


class MetadataEvidenceTests(unittest.TestCase):
    """Evidence projection must survive copy-hostile SDK quant descriptors."""

    def test_fixture_reproduces_the_board_asdict_failure(self):
        from samples._shared.runtime_meta import RuntimeMetadata

        meta = RuntimeMetadata.from_runtime(_SingleModelRuntime(), "det")
        with self.assertRaises(TypeError) as ctx:
            asdict(meta)
        self.assertIn("cannot pickle", str(ctx.exception))
        with self.assertRaises(TypeError):
            copy.deepcopy(meta.output_quants["out"])

    def test_projection_survives_deepcopy_hostile_quants_and_dumps_strictly(self):
        from samples._shared.runtime_meta import RuntimeMetadata, metadata_evidence

        meta = RuntimeMetadata.from_runtime(_SingleModelRuntime(), "det")
        view = metadata_evidence(meta)
        dumped = json.dumps(view, allow_nan=False)  # must not raise
        self.assertIsInstance(dumped, str)

        out = view["output_quants"]["out"]
        self.assertEqual(out["quant_type"], "SCALE")
        self.assertEqual(out["scale"], [0.25, 0.5])
        self.assertEqual(out["zero_point"], [3, -7])
        self.assertEqual(out["axis"], 3)
        aux = view["output_quants"]["aux"]
        self.assertEqual(aux["quant_type"], "NONE")
        self.assertEqual(aux["scale"], 0.0)
        self.assertEqual(aux["zero_point"], 0)

    def test_projection_preserves_tensor_facts_and_field_coverage(self):
        from samples._shared.runtime_meta import RuntimeMetadata, metadata_evidence

        meta = RuntimeMetadata.from_runtime(_SingleModelRuntime(), "det")
        view = metadata_evidence(meta)
        self.assertEqual(view["model_name"], "det")
        self.assertEqual(view["model_names"], ["det"])
        self.assertEqual(view["input_names"], ["x"])
        self.assertEqual(view["input_shapes"], {"x": [1, 3, 8, 8]})
        self.assertEqual(view["output_shapes"]["out"], [1, 8, 8, 255])
        self.assertEqual(view["input_dtypes"], {"x": "nv12"})
        self.assertEqual(view["output_dtypes"], {"out": "int8", "aux": "float32"})
        self.assertEqual(view["input_strides"], {"x": [1, 1, 8, 1]})
        self.assertEqual(view["output_strides"], {"out": [2040, 255, 1, 1]})
        self.assertEqual(
            set(view),
            {
                "model_name",
                "input_names",
                "input_shapes",
                "output_names",
                "output_shapes",
                "input_dtypes",
                "output_dtypes",
                "model_names",
                "input_strides",
                "output_strides",
                "output_quants",
                "input_quants",
                "output_semantics",
            },
        )
        self.assertIsNone(view["output_semantics"])

    def test_projection_has_no_side_effects_on_metadata_or_sdk_objects(self):
        from samples._shared.runtime_meta import RuntimeMetadata, metadata_evidence

        meta = RuntimeMetadata.from_runtime(_SingleModelRuntime(), "det")
        descriptor = meta.output_quants["out"]
        scale_before = descriptor.scale.copy()
        zero_before = descriptor.zero_point.copy()
        view = metadata_evidence(meta)
        # The metadata still owns the very same SDK objects, untouched.
        self.assertIs(meta.output_quants["out"], descriptor)
        np.testing.assert_array_equal(descriptor.scale, scale_before)
        np.testing.assert_array_equal(descriptor.zero_point, zero_before)
        self.assertEqual(descriptor.scale.dtype, scale_before.dtype)
        # The projection is a fresh structure; editing it cannot leak back.
        view["output_quants"]["out"]["scale"][0] = 999.0
        self.assertEqual(float(descriptor.scale[0]), 0.25)

    def test_projection_keeps_extra_public_quant_attributes(self):
        from samples._shared.runtime_meta import RuntimeMetadata, metadata_evidence

        runtime = _MultiModelRuntime()
        runtime.output_quants = {
            **runtime.output_quants,
            "rec": {
                **runtime.output_quants["rec"],
                "out_rec": _BoardQuantParams(scale=0.5, zero_point=3, tensor_name="conv_out"),
            },
        }
        view = metadata_evidence(RuntimeMetadata.from_runtime(runtime, "rec"))
        self.assertEqual(view["output_quants"]["out_rec"]["tensor_name"], "conv_out")
        json.dumps(view, allow_nan=False)

    def test_projection_accepts_mapping_metadata_for_host_seams(self):
        from samples._shared.runtime_meta import metadata_evidence

        view = metadata_evidence(
            {
                "model_name": "m",
                "input_shapes": {"x": (1, 3, 4, 4)},
                "output_quants": {
                    "out": _BoardQuantParams(scale=np.array([0.1], np.float32), axis=1)
                },
            }
        )
        self.assertEqual(view["input_shapes"], {"x": [1, 3, 4, 4]})
        # float32 widens exactly; the projection must not round-trip through text.
        self.assertEqual(view["output_quants"]["out"]["scale"], [np.float32(0.1).item()])
        self.assertEqual(view["output_quants"]["out"]["axis"], 1)
        json.dumps(view, allow_nan=False)

    def test_projection_rejects_unknown_objects_instead_of_stringifying(self):
        from samples._shared.runtime_meta import RuntimeMetadata, metadata_evidence

        class _Opaque:
            pass

        runtime = _SingleModelRuntime()
        runtime.output_quants = {
            **runtime.output_quants,
            "det": {**runtime.output_quants["det"], "out": _Opaque()},
        }
        with self.assertRaises(TypeError) as ctx:
            metadata_evidence(RuntimeMetadata.from_runtime(runtime, "det"))
        self.assertIn("_Opaque", str(ctx.exception))
        self.assertNotIn("str(", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
