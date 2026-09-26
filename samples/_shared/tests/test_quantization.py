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

"""Host tests for the declared output transform chain (Phase 1.5 H1).

The dequantization math is checked against hand-computed values following the
semantics ported from the delivery branches' ``utils/py_utils/postprocess.py``
(rdk_s @ 380e1a2).  Nothing here executes a board runtime.
"""
from __future__ import annotations

import unittest

import numpy as np


class _EnumLike:
    def __init__(self, name: str) -> None:
        self.name = name


class _QuantInfo:
    def __init__(self, *, scale, zero_point, axis=1, quant_type="SCALE") -> None:
        self.quant_type = _EnumLike(quant_type)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.zero_point = np.asarray(zero_point, dtype=np.float32)
        self.axis = axis


class DequantTensorTests(unittest.TestCase):
    def test_per_tensor_dequantization_matches_source_semantics(self):
        from samples._shared.quantization import dequantize_tensor

        q = np.array([[-10, 0, 10], [20, 30, 40]], dtype=np.int8)
        info = _QuantInfo(scale=0.125, zero_point=3)
        out = dequantize_tensor(q, info)
        np.testing.assert_allclose(
            out, (q.astype(np.float32) - 3.0) * 0.125, rtol=0, atol=0
        )
        self.assertEqual(out.dtype, np.float32)

    def test_per_channel_dequantization_broadcasts_along_axis(self):
        from samples._shared.quantization import dequantize_tensor

        q = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
        # axis=1 with three channels: one scale/zero-point per column.
        info = _QuantInfo(scale=[0.5, 1.0, 2.0], zero_point=[1, 0, -1], axis=1)
        out = dequantize_tensor(q, info)
        expected = np.empty((2, 3), dtype=np.float32)
        expected[:, 0] = (q[:, 0].astype(np.float32) - 1) * 0.5
        expected[:, 1] = (q[:, 1].astype(np.float32) - 0) * 1.0
        expected[:, 2] = (q[:, 2].astype(np.float32) + 1) * 2.0
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-6)

    def test_per_channel_single_zero_point_is_broadcast(self):
        from samples._shared.quantization import dequantize_tensor

        q = np.array([[2, 4]], dtype=np.int8)
        info = _QuantInfo(scale=[1.0, 3.0], zero_point=[7], axis=1)
        out = dequantize_tensor(q, info)
        # Affine values are (q - zero_point) * scale.  The source helper
        # incorrectly discarded a scalar nonzero zero-point on this path.
        np.testing.assert_array_equal(out, [[-5.0, -9.0]])

    def test_scalar_offset_can_change_segmentation_argmax(self):
        from samples._shared.quantization import dequantize_tensor

        logits = np.array([[[[2, 4]]]], dtype=np.int16)
        info = _QuantInfo(scale=[1.0, 3.0], zero_point=[7], axis=-1)
        decoded = dequantize_tensor(logits, info)
        # With the offset the scores are [-5, -9], so class 0 wins.
        # Ignoring it gives [2, 12], which incorrectly selects class 1.
        self.assertEqual(int(np.argmax(decoded, axis=-1).item()), 0)
        np.testing.assert_array_equal(logits, [[[[2, 4]]]])

    def test_scalar_zero_point_broadcasts_on_non_last_axis(self):
        from samples._shared.quantization import dequantize_tensor

        q = np.array([[[2, 4], [6, 8]]], dtype=np.int32)
        info = _QuantInfo(scale=[0.5, 2.0], zero_point=-2, axis=1)
        np.testing.assert_array_equal(
            dequantize_tensor(q, info), [[[2, 3], [16, 20]]]
        )

    def test_empty_per_channel_zero_point_is_symmetric(self):
        from samples._shared.quantization import dequantize_tensor

        q = np.array([[2, 4]], dtype=np.int16)
        info = _QuantInfo(scale=[1.0, 3.0], zero_point=[], axis=1)
        np.testing.assert_array_equal(dequantize_tensor(q, info), [[2, 12]])

    def test_empty_zero_point_becomes_zero(self):
        from samples._shared.quantization import dequantize_tensor

        q = np.array([5], dtype=np.int8)
        info = _QuantInfo(scale=2.0, zero_point=[])
        out = dequantize_tensor(q, info)
        np.testing.assert_allclose(out, [10.0], rtol=0, atol=0)

    def test_non_scale_quant_type_passes_through_unchanged(self):
        from samples._shared.quantization import dequantize_tensor

        q = np.array([7, 8], dtype=np.int8)
        info = _QuantInfo(scale=0.5, zero_point=0, quant_type="NON_SCALE")
        self.assertIs(dequantize_tensor(q, info), q)


class DequantOutputsTests(unittest.TestCase):
    def test_mapping_form_dequantizes_each_output(self):
        from samples._shared.quantization import dequantize_outputs

        outputs = {"a": np.array([4], dtype=np.int8), "b": np.array([6], dtype=np.int8)}
        quants = {
            "a": _QuantInfo(scale=1.0, zero_point=0),
            "b": _QuantInfo(scale=0.5, zero_point=2),
        }
        out = dequantize_outputs(outputs, quants)
        np.testing.assert_allclose(out["a"], [4.0], rtol=0, atol=0)
        np.testing.assert_allclose(out["b"], [2.0], rtol=0, atol=0)


class ApplyTransformTests(unittest.TestCase):
    def test_raw_f32_passthrough(self):
        from samples._shared.quantization import apply_output_transform

        values = {"prob": np.array([1.0, 2.0], dtype=np.float32)}
        out = apply_output_transform("raw_f32", values)
        self.assertIs(out["prob"], values["prob"])

    def test_raw_f32_rejects_integer_dtype(self):
        from samples._shared.quantization import (
            OutputTransformError,
            apply_output_transform,
        )

        with self.assertRaises(OutputTransformError):
            apply_output_transform(
                "raw_f32", {"prob": np.zeros(3, dtype=np.int8)}
            )

    def test_raw_f32_ignores_vestigial_quant_descriptor(self):
        # Contract refinement driven by board evidence (X5 smoke, 2026-09-21):
        # published X5 artifacts ship F32 outputs that still carry a compiler
        # quant descriptor.  The values are final floats — the descriptor is
        # neither applied nor an error, matching legacy consumers.
        from samples._shared.quantization import apply_output_transform

        values = {"prob": np.array([1.0, 2.0], dtype=np.float32)}
        out = apply_output_transform(
            "raw_f32",
            values,
            {"prob": _QuantInfo(scale=0.5, zero_point=0)},
        )
        self.assertIs(out["prob"], values["prob"])

    def test_dequant_requires_descriptor_for_every_output(self):
        from samples._shared.quantization import (
            OutputTransformError,
            apply_output_transform,
        )

        with self.assertRaises(OutputTransformError):
            apply_output_transform(
                "dequant", {"prob": np.zeros(3, dtype=np.int8)}
            )

    def test_dequant_returns_float32(self):
        from samples._shared.quantization import apply_output_transform

        out = apply_output_transform(
            "dequant",
            {"prob": np.array([10], dtype=np.int8)},
            {"prob": _QuantInfo(scale=0.1, zero_point=0)},
        )
        np.testing.assert_allclose(out["prob"], [1.0], rtol=0, atol=0)
        self.assertEqual(out["prob"].dtype, np.float32)

    def test_unknown_transform_name_is_rejected(self):
        from samples._shared.quantization import (
            OutputTransformError,
            validate_output_transform,
        )

        with self.assertRaises(OutputTransformError):
            validate_output_transform("maybe_dequant")
        self.assertEqual(validate_output_transform("raw_f32"), "raw_f32")


if __name__ == "__main__":
    unittest.main()


class DequantizationPrecisionTests(unittest.TestCase):
    def test_explicit_float64_retains_int32_differences_default_unchanged(self):
        from types import SimpleNamespace
        from samples._shared.quantization import dequantize_tensor
        q=SimpleNamespace(quant_type=SimpleNamespace(name='SCALE'),axis=0,
                          scale=np.array([1.],np.float32),zero_point=np.array([0],np.int32))
        raw=np.array([2**25,2**25+1],np.int32)
        default=dequantize_tensor(raw,q)
        precise=dequantize_tensor(raw,q,dtype='float64')
        self.assertEqual(default.dtype,np.float32)
        self.assertEqual(precise.dtype,np.float64)
        self.assertEqual(np.argmax(default),0)
        self.assertEqual(np.argmax(precise),1)
        np.testing.assert_array_equal(raw,[2**25,2**25+1])
        with self.assertRaises(ValueError):dequantize_tensor(raw,q,dtype='int32')
