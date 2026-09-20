"""Host tests for the finite PaddleOCR model and tensor contracts."""

from __future__ import annotations

import unittest


def _metadata_x5(stage: str = "det", **overrides):
    from samples.vision.paddle_ocr.runtime.python.model_binding import RuntimeMetadata

    if stage == "det":
        values = {
            "model_name": "en_PP-OCRv3_det_infer-deploy_640x640_nv12",
            "input_names": ["x"],
            "input_shapes": {"x": (1, 3, 640, 640)},
            "input_dtypes": {"x": "hbDNNDataType.NV12"},
            "output_names": ["sigmoid_0.tmp_0"],
            "output_shapes": {"sigmoid_0.tmp_0": (1, 1, 640, 640)},
            "output_dtypes": {"sigmoid_0.tmp_0": "hbDNNDataType.F32"},
        }
    else:
        values = {
            "model_name": "en_PP-OCRv3_rec_infer-deploy_48x320_rgb_NCHW",
            "input_names": ["x"],
            "input_shapes": {"x": (1, 3, 48, 320)},
            "input_dtypes": {"x": "hbDNNDataType.F32"},
            "output_names": ["softmax_2.tmp_0"],
            "output_shapes": {"softmax_2.tmp_0": (1, 40, 97, 1)},
            "output_dtypes": {"softmax_2.tmp_0": "hbDNNDataType.F32"},
        }
    values.update(overrides)
    return RuntimeMetadata.from_mapping(values)


def _metadata_s100(stage: str = "det", **overrides):
    from samples.vision.paddle_ocr.runtime.python.model_binding import RuntimeMetadata

    if stage == "det":
        values = {
            "model_name": "PP-OCRv6_det_infer-deploy_640x640_nv12",
            "input_names": ["x_y", "x_uv"],
            "input_shapes": {
                "x_y": (1, 640, 640, 1),
                "x_uv": (1, 320, 320, 2),
            },
            "input_dtypes": {"x_y": "hbDNNDataType.U8", "x_uv": "hbDNNDataType.U8"},
            "output_names": ["fetch_name_0"],
            "output_shapes": {"fetch_name_0": (1, 1, 640, 640)},
            "output_dtypes": {"fetch_name_0": "hbDNNDataType.F32"},
        }
    else:
        values = {
            "model_name": "PP-OCRv6_rec_infer-deploy_48x320_rgb",
            "input_names": ["x"],
            "input_shapes": {"x": (1, 3, 48, 320)},
            "input_dtypes": {"x": "hbDNNDataType.F32"},
            "output_names": ["fetch_name_0"],
            "output_shapes": {"fetch_name_0": (1, 40, 18710)},
            "output_dtypes": {"fetch_name_0": "hbDNNDataType.F32"},
        }
    values.update(overrides)
    return RuntimeMetadata.from_mapping(values)


class BindingTests(unittest.TestCase):
    def test_list_contains_only_the_two_audited_pairs(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import list_available_pairs

        pairs = list_available_pairs()
        self.assertEqual({pair.target for pair in pairs}, {"x5", "s100"})
        self.assertEqual(len(pairs), 2)
        self.assertEqual(
            pairs[0].detector_asset,
            "x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin",
        )
        self.assertEqual(
            pairs[1].recognizer_asset,
            "s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm",
        )

    def test_custom_paths_require_both_qualified_asset_references(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import (
            BindingError,
            resolve_pair,
        )

        with self.assertRaises(BindingError):
            resolve_pair(
                "x5",
                det_model_path="det.bin",
                rec_model_path="rec.bin",
            )
        with self.assertRaises(BindingError):
            resolve_pair(
                "x5",
                det_asset_id="x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin",
                det_model_path="det.bin",
            )
        with self.assertRaises(BindingError):
            resolve_pair(
                "x5",
                det_asset_id="x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin",
                rec_asset_id="x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin",
                det_model_path="same.bin",
                rec_model_path="same.bin",
            )

    def test_mixed_target_pair_is_rejected(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import (
            BindingError,
            resolve_pair,
        )

        with self.assertRaises(BindingError):
            resolve_pair(
                "auto",
                det_asset_id="x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin",
                rec_asset_id="s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm",
            )

    def test_actual_x5_and_s100_metadata_bind_to_exact_names(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import (
            bind_stage,
            resolve_pair,
        )

        x5 = resolve_pair("x5")
        x5_det = bind_stage(x5, "detector", _metadata_x5())
        x5_rec = bind_stage(x5, "recognizer", _metadata_x5("rec"))
        self.assertEqual(x5_det.input_names, ("x",))
        self.assertEqual(x5_det.output_name, "sigmoid_0.tmp_0")
        self.assertEqual(x5_rec.output_shape, (1, 40, 97, 1))

        s100 = resolve_pair("s100")
        s_det = bind_stage(s100, "detector", _metadata_s100())
        s_rec = bind_stage(s100, "recognizer", _metadata_s100("rec"))
        self.assertEqual(s_det.input_names, ("x_y", "x_uv"))
        self.assertEqual(s_det.output_name, "fetch_name_0")
        self.assertEqual(s_rec.output_shape, (1, 40, 18710))

    def test_metadata_mismatch_and_missing_dtype_are_rejected(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import (
            MetadataMismatchError,
            bind_stage,
            resolve_pair,
        )

        pair = resolve_pair("x5")
        with self.assertRaises(MetadataMismatchError):
            bind_stage(pair, "detector", _metadata_x5(input_names=["wrong"]))
        with self.assertRaises(MetadataMismatchError):
            bind_stage(
                pair,
                "recognizer",
                _metadata_x5("rec", output_shapes={"softmax_2.tmp_0": (1, 40, 96, 1)}),
            )
        with self.assertRaises(MetadataMismatchError):
            bind_stage(
                pair,
                "recognizer",
                _metadata_x5("rec", output_dtypes={}),
            )

    def test_direct_runtime_metadata_has_safe_optional_mappings(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import (
            RuntimeMetadata,
        )

        metadata = RuntimeMetadata(
            model_name="model",
            input_names=("x",),
            input_shapes={"x": (1, 3, 640, 640)},
            output_names=("out",),
            output_shapes={"out": (1, 1, 640, 640)},
            input_dtypes={"x": "nv12"},
            output_dtypes={"out": "float32"},
        )
        self.assertEqual(metadata.output_quants, {})
        self.assertEqual(metadata.model_names, ())
        self.assertEqual(metadata.input_strides, {})
        self.assertEqual(metadata.output_strides, {})

    def test_bound_stage_validates_physical_runtime_tensors(self):
        import numpy as np

        from samples.vision.paddle_ocr.runtime.python.model_binding import (
            bind_stage,
            resolve_pair,
            validate_stage_inputs,
            validate_stage_output,
        )

        pair = resolve_pair("x5")
        binding = bind_stage(pair, "detector", _metadata_x5())
        inputs = validate_stage_inputs(
            binding,
            {"x": np.zeros((1, 960, 640, 1), dtype=np.uint8)},
        )
        outputs = validate_stage_output(
            binding,
            {"sigmoid_0.tmp_0": np.zeros((1, 1, 640, 640), dtype=np.float32)},
        )
        self.assertEqual(inputs["x"].shape, (1, 960, 640, 1))
        self.assertEqual(outputs["sigmoid_0.tmp_0"].dtype, np.float32)

    def test_vocabulary_policies_keep_fixed_x5_and_hashed_s100_table(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import (
            X5_ALPHABET,
            list_available_pairs,
        )

        pairs = {pair.target: pair for pair in list_available_pairs()}
        x5_tokens = pairs["x5"].vocabulary.load_tokens()
        self.assertEqual(len(X5_ALPHABET), 96)
        self.assertEqual(len(x5_tokens), 97)
        self.assertEqual(x5_tokens[0], "blank")
        self.assertEqual(x5_tokens[-2:], (" ", " "))

        s100_tokens = pairs["s100"].vocabulary.load_tokens()
        self.assertEqual(len(s100_tokens), 18710)
        self.assertEqual(s100_tokens[0], "blank")
        self.assertEqual(s100_tokens[-1], " ")
        self.assertIn("照", s100_tokens)

    def test_all_present_quantization_descriptors_are_rejected_for_each_stage(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import (
            MetadataMismatchError,
            bind_stage,
            list_available_pairs,
        )

        for pair in list_available_pairs():
            for stage in ("detector", "recognizer"):
                contract = getattr(pair, stage)
                for values in (
                    {contract.output_name: [0.5]},
                    {contract.output_name: [0.5, 0.75]},
                    {contract.output_name: object()},
                ):
                    metadata = {
                        "model_name": contract.model_name,
                        "input_names": list(contract.input_names),
                        "input_shapes": dict(contract.input_shapes),
                        "input_dtypes": dict(contract.input_dtypes),
                        "output_names": [contract.output_name],
                        "output_shapes": {contract.output_name: contract.output_shape},
                        "output_dtypes": {contract.output_name: "float32"},
                        # The F32 stage contract must reject any present
                        # output_quants descriptor (H1 raw_f32 discipline).
                        "output_quants": values,
                    }
                    with self.assertRaises(MetadataMismatchError):
                        bind_stage(pair, stage, metadata)

    def test_multi_model_runtime_requires_explicit_model_selection(self):
        from samples._shared.runtime_meta import (
            MetadataMismatchError as SharedMetadataMismatchError,
        )
        from samples.vision.paddle_ocr.runtime.python.model_binding import (
            RuntimeMetadata,
        )

        MetadataMismatchError = SharedMetadataMismatchError

        class _TwoModelRuntime:
            model_names = ["det_model", "rec_model"]
            input_names = {
                "det_model": ["x"],
                "rec_model": ["x"],
            }
            input_shapes = {
                "det_model": {"x": (1, 3, 640, 640)},
                "rec_model": {"x": (1, 3, 48, 320)},
            }
            input_dtypes = {
                "det_model": {"x": "NV12"},
                "rec_model": {"x": "F32"},
            }
            output_names = {
                "det_model": ["sigmoid_0.tmp_0"],
                "rec_model": ["softmax_2.tmp_0"],
            }
            output_shapes = {
                "det_model": {"sigmoid_0.tmp_0": (1, 1, 640, 640)},
                "rec_model": {"softmax_2.tmp_0": (1, 40, 97, 1)},
            }
            output_dtypes = {
                "det_model": {"sigmoid_0.tmp_0": "F32"},
                "rec_model": {"softmax_2.tmp_0": "F32"},
            }

        # H3: several models are never silently reduced to model_names[0].
        with self.assertRaises(MetadataMismatchError):
            RuntimeMetadata.from_runtime(_TwoModelRuntime())
        selected = RuntimeMetadata.from_runtime(_TwoModelRuntime(), "rec_model")
        self.assertEqual(selected.model_name, "rec_model")
        self.assertEqual(selected.model_names, ("det_model", "rec_model"))
        self.assertEqual(selected.input_shapes, {"x": (1, 3, 48, 320)})
        self.assertEqual(
            selected.output_shapes, {"softmax_2.tmp_0": (1, 40, 97, 1)}
        )
        with self.assertRaises(MetadataMismatchError):
            RuntimeMetadata.from_runtime(_TwoModelRuntime(), "missing_model")


if __name__ == "__main__":
    unittest.main()
