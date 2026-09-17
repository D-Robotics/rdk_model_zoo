"""Host seams for the reviewed YOLO26 direct LTRB protocol."""

import unittest
import types

import numpy as np

from samples.vision.ultralytics_yolo.runtime.python.decode import (
    DecodeError,
    decode_dfl,
    decode_ltrb,
)
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    BindingError,
    DFLDetectionContract,
    LTRBDetectionContract,
    ModelSelection,
    RuntimeMetadata,
    bind_model,
)
from samples.vision.ultralytics_yolo.runtime.python.tensor_io import InputBinding
from samples.vision.ultralytics_yolo.runtime.python.yolo26_det import (
    YOLO26Detect,
    YOLO26DetectConfig,
)
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import resolve_platform


def _semantic_ltrb_outputs(input_size=(64, 64), *, classes=1,
                           box=(1.0, 2.0, 3.0, 4.0)):
    outputs = {}
    for stride in (8, 16, 32):
        height, width = input_size[0] // stride, input_size[1] // stride
        cls = np.full((1, height, width, classes), -20.0, dtype=np.float32)
        boxes = np.zeros((1, height, width, 4), dtype=np.float32)
        if stride == 8:
            cls[0, 0, 0, 0] = 4.0
            boxes[0, 0, 0] = np.asarray(box, dtype=np.float32)
        outputs[f"cls_{stride}"] = cls
        outputs[f"box_{stride}"] = boxes
    return outputs


def _metadata(platform, *, bad_box_shape=False, output_dtype=np.float32):
    if platform == "x5":
        input_names = ("images",)
        input_shapes = {"images": (1, 3, 64, 64)}
        input_dtypes = {"images": "hbDNNDataType.NV12"}
        output_names = ("output0", "580", "594", "602", "616", "624")
        output_shapes = {
            "output0": (1, 8, 8, 1),
            "580": (1, 8, 8, 4),
            "594": (1, 4, 4, 1),
            "602": (1, 4, 4, 4),
            "616": (1, 2, 2, 1),
            "624": (1, 2, 2, 4),
        }
    else:
        input_names = ("images_y", "images_uv")
        input_shapes = {
            "images_y": (1, 64, 64, 1),
            "images_uv": (1, 32, 32, 2),
        }
        input_dtypes = {
            "images_y": "hbDNNDataType.U8",
            "images_uv": "hbDNNDataType.U8",
        }
        output_names = ("output0", "609", "623", "631", "645", "653")
        output_shapes = {
            "output0": (1, 8, 8, 1),
            "609": (1, 8, 8, 4),
            "623": (1, 4, 4, 1),
            "631": (1, 4, 4, 4),
            "645": (1, 2, 2, 1),
            "653": (1, 2, 2, 4),
        }
    if bad_box_shape:
        first_box = "580" if platform == "x5" else "609"
        output_shapes[first_box] = (1, output_shapes[first_box][1],
                                    output_shapes[first_box][2], 64)
    return RuntimeMetadata(
        model_name="yolo26",
        input_names=input_names,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_names=output_names,
        output_shapes=output_shapes,
        output_dtypes={name: output_dtype for name in output_names},
    )


class LTRBBindingTests(unittest.TestCase):
    def test_direct_offsets_are_not_interpreted_as_dfl_bins(self):
        contract = LTRBDetectionContract(classes=1, strides=(8, 16, 32))
        expected = np.array([[-4.0, -12.0, 28.0, 36.0]], dtype=np.float32)
        boxes, scores, ids = decode_ltrb(
            _semantic_ltrb_outputs(), contract, input_size=(64, 64),
            score_thres=0.5, nms_thres=0.5,
        )
        np.testing.assert_allclose(boxes, expected)
        np.testing.assert_allclose(scores, [1.0 / (1.0 + np.exp(-4.0))])
        np.testing.assert_array_equal(ids, [0])

        dfl = DFLDetectionContract(classes=1, reg_bins=1, strides=(8, 16, 32))
        dfl_boxes, _, _ = decode_dfl(
            _semantic_ltrb_outputs(), dfl, input_size=(64, 64),
            score_thres=0.5, nms_thres=0.5,
        )
        self.assertFalse(np.allclose(boxes, dfl_boxes))

    def test_large_logits_keep_raw_class_order(self):
        contract = LTRBDetectionContract(classes=2)
        outputs = _semantic_ltrb_outputs(classes=2)
        outputs["cls_8"][0, 0, 0] = [18.0, 20.0]
        boxes, scores, ids = decode_ltrb(
            outputs, contract, input_size=(64, 64),
            score_thres=0.5, nms_thres=0.5,
        )
        self.assertEqual(ids.tolist(), [1])
        self.assertEqual(scores.tolist(), [1.0])

    def test_dfl_large_logits_keep_raw_class_order(self):
        """DFL keeps source class identity when sigmoid saturates in float32."""
        contract = DFLDetectionContract(
            classes=2, reg_bins=16, strides=(8, 16, 32), nms="classwise")
        outputs = {}
        for stride in contract.strides:
            height, width = 64 // stride, 64 // stride
            cls = np.full((1, height, width, 2), -20.0, dtype=np.float32)
            box = np.zeros((1, height, width, 64), dtype=np.float32)
            if stride == 8:
                cls[0, 0, 0] = [18.0, 20.0]
            outputs[f"cls_{stride}"] = cls
            outputs[f"box_{stride}"] = box

        _, scores, ids = decode_dfl(
            outputs, contract, input_size=(64, 64),
            score_thres=0.5, nms_thres=0.5)
        self.assertEqual(ids.tolist(), [1])
        self.assertEqual(scores.tolist(), [1.0])

    def test_observed_x5_and_s_names_bind_to_ltrb_roles(self):
        contract = LTRBDetectionContract(classes=1)
        for platform in ("x5", "s100"):
            with self.subTest(platform=platform):
                binding = bind_model(
                    ModelSelection("fixture.hbm", target=platform,
                                   contract=contract),
                    _metadata(platform),
                )
                self.assertEqual(binding.output_name("cls", 8), "output0")
                self.assertEqual(
                    binding.output_name("box", 8),
                    "580" if platform == "x5" else "609",
                )
                self.assertEqual(binding.contract.box_channels, 4)

    def test_classification_threshold_uses_declared_domain(self):
        raw_threshold = np.float32(-np.log(1.0 / 0.25 - 1.0))
        raw_below = np.nextafter(raw_threshold, -np.inf)
        raw_above = np.nextafter(raw_threshold, np.inf)

        def outputs_for(value):
            outputs = {}
            for stride in (8, 16, 32):
                height, width = 64 // stride, 64 // stride
                cls = np.full((1, height, width, 1), -20.0, dtype=np.float32)
                box = np.zeros((1, height, width, 4), dtype=np.float32)
                if stride == 8:
                    cls[0, 0, 0, 0] = value
                outputs[f"cls_{stride}"] = cls
                outputs[f"box_{stride}"] = box
            return outputs

        logits = DFLDetectionContract(
            classes=1, reg_bins=1, strides=(8, 16, 32),
            classification="logits", nms="classwise")
        _, _, ids = decode_dfl(
            outputs_for(raw_below), logits, input_size=(64, 64),
            score_thres=0.25, nms_thres=0.5)
        self.assertEqual(ids.size, 0)
        _, _, ids = decode_dfl(
            outputs_for(raw_above), logits, input_size=(64, 64),
            score_thres=0.25, nms_thres=0.5)
        self.assertEqual(ids.tolist(), [0])

        probabilities = DFLDetectionContract(
            classes=1, reg_bins=1, strides=(8, 16, 32),
            classification="probabilities", nms="classwise")
        probability_below = np.nextafter(np.float32(0.25), -np.inf)
        probability_outputs = outputs_for(0.0)
        for stride in probabilities.strides:
            probability_outputs[f"cls_{stride}"].fill(0.0)
        probability_outputs["cls_8"][0, 0, 0, 0] = probability_below
        _, _, ids = decode_dfl(
            probability_outputs, probabilities, input_size=(64, 64),
            score_thres=0.25, nms_thres=0.5)
        self.assertEqual(ids.size, 0)
        probability_outputs["cls_8"][0, 0, 0, 0] = np.float32(0.25)
        _, _, ids = decode_dfl(
            probability_outputs, probabilities, input_size=(64, 64),
            score_thres=0.25, nms_thres=0.5)
        self.assertEqual(ids.tolist(), [0])

    def test_wrong_channel_dtype_and_nonfinite_outputs_are_rejected(self):
        contract = LTRBDetectionContract(classes=1)
        selection = ModelSelection("fixture.hbm", target="x5",
                                   contract=contract)
        with self.assertRaises(BindingError):
            bind_model(selection, _metadata("x5", bad_box_shape=True))
        with self.assertRaises(BindingError):
            bind_model(selection, _metadata("x5", output_dtype=np.int8))

        raw = _metadata("x5")
        quantized = RuntimeMetadata(
            model_name=raw.model_name,
            input_names=raw.input_names,
            input_shapes=raw.input_shapes,
            input_dtypes=raw.input_dtypes,
            output_names=raw.output_names,
            output_shapes=raw.output_shapes,
            output_dtypes=raw.output_dtypes,
            output_quantization={"output0": {"scale": 1.0}},
        )
        with self.assertRaises(BindingError):
            bind_model(selection, quantized)

        binding = bind_model(selection, _metadata("x5"))
        physical = {
            "yolo26": {
                name: np.zeros(shape, dtype=np.float32)
                for name, shape in binding.metadata.output_shapes.items()
            }
        }
        physical["yolo26"]["output0"][0, 0, 0, 0] = np.nan
        with self.assertRaises(BindingError):
            binding.read_outputs(physical)

        with self.assertRaises(DecodeError):
            decode_ltrb(
                {key: value.astype(np.int8)
                 for key, value in _semantic_ltrb_outputs().items()},
                contract, input_size=(64, 64), nms_thres=0.5,
            )

    def test_injected_runner_executes_once_and_preserves_tuple_result(self):
        adapter = InputBinding(
            model_name="fixture",
            roles={"image": "images"},
            shapes={"images": (1, 3, 64, 64)},
            dtypes={"images": np.dtype(np.uint8)},
            packed=True,
            height=64,
            width=64,
        )
        contract = LTRBDetectionContract(classes=1)

        class FakeRunner:
            input_adapter = adapter
            input_size = (64, 64)

            def __init__(self):
                self.calls = 0

            def __call__(self, tensors):
                self.calls += 1
                assert tensors["fixture"]["images"].size == 64 * 64 * 3 // 2
                return _semantic_ltrb_outputs()

        runner = FakeRunner()
        detector = YOLO26Detect(
            YOLO26DetectConfig(
                model_path="fixture.hbm",
                classes_num=1,
                strides=[8, 16, 32],
                score_thres=0.5,
                nms_thres=0.5,
                contract=contract,
            ),
            runner=runner,
        )
        result = detector.predict(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertEqual(runner.calls, 1)
        np.testing.assert_allclose(result.boxes_xyxy,
                                   [[0.0, 0.0, 28.0, 36.0]])
        with self.assertRaises(BindingError):
            detector.set_scheduling_params(priority=0)

    def test_factory_binds_physical_outputs_before_ltrb_execution(self):
        metadata = _metadata("x5")
        role_names = {
            "cls_8": "output0", "box_8": "580",
            "cls_16": "594", "box_16": "602",
            "cls_32": "616", "box_32": "624",
        }
        semantic = _semantic_ltrb_outputs()

        class FakeModel:
            model_names = [metadata.model_name]
            input_names = {metadata.model_name: list(metadata.input_names)}
            input_shapes = {metadata.model_name: dict(metadata.input_shapes)}
            input_dtypes = {metadata.model_name: dict(metadata.input_dtypes)}
            output_names = {metadata.model_name: list(metadata.output_names)}
            output_shapes = {metadata.model_name: dict(metadata.output_shapes)}
            output_dtypes = {metadata.model_name: {
                name: "hbDNNDataType.F32" for name in metadata.output_names
            }}

            def run(self, tensors):
                del tensors
                return {metadata.model_name: {
                    name: semantic[role]
                    for role, name in role_names.items()
                }}

        model = FakeModel()
        runtime = types.SimpleNamespace(
            HB_HBMRuntime=lambda path: model,
        )
        detector = YOLO26Detect(
            YOLO26DetectConfig(
                model_path="fixture.bin",
                platform=resolve_platform("x5"),
                classes_num=1,
                score_thres=0.5,
                nms_thres=0.5,
            ),
            runtime_loader=lambda: runtime,
        )
        result = detector.predict(np.zeros((64, 64, 3), dtype=np.uint8))
        np.testing.assert_allclose(result.boxes_xyxy,
                                   [[0.0, 0.0, 28.0, 36.0]])


if __name__ == "__main__":
    unittest.main()
