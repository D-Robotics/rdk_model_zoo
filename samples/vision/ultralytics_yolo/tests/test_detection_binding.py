"""Host tests for the detector's explicit model boundary and geometry."""

import unittest

import numpy as np


from samples.vision.ultralytics_yolo.runtime.python.geometry import (
    ImageTransform,
    inverse_boxes,
    resize_with_transform,
)
from samples.vision.ultralytics_yolo.runtime.python.decode import DecodeError, decode_dfl
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    DFLDetectionContract,
    ModelSelection,
    RuntimeMetadata,
    bind_model,
    BindingError,
)
from samples.vision.ultralytics_yolo.runtime.python.model_runner import build_runner
from samples.vision.ultralytics_yolo.runtime.python.tensor_io import (
    InputBinding,
    normalize_dtype,
)
from samples.vision.ultralytics_yolo.runtime.python.yolo_detect import (
    YoloDetect,
    YoloDetectConfig,
)


def _metadata(*, output_dtype=np.float32, output_shapes=None):
    """Build a small runtime fixture with deliberately named tensors."""
    output_shapes = output_shapes or {
        "head_cls_8": (1, 8, 12, 1),
        "head_box_8": (1, 8, 12, 64),
        "head_cls_16": (1, 4, 6, 1),
        "head_box_16": (1, 4, 6, 64),
        "head_cls_32": (1, 2, 3, 1),
        "head_box_32": (1, 2, 3, 64),
    }
    return RuntimeMetadata(
        model_name="detector",
        input_names=("nv12",),
        input_shapes={"nv12": (1, 3, 64, 96)},
        input_dtypes={"nv12": np.dtype(np.uint8)},
        output_names=tuple(output_shapes),
        output_shapes=output_shapes,
        output_dtypes={name: np.dtype(output_dtype) for name in output_shapes},
    )


def _contract():
    return DFLDetectionContract(
        classes=1,
        reg_bins=16,
        strides=(8, 16, 32),
        input_roles={"image": "nv12"},
        output_roles={
            "cls_8": "head_cls_8",
            "box_8": "head_box_8",
            "cls_16": "head_cls_16",
            "box_16": "head_box_16",
            "cls_32": "head_cls_32",
            "box_32": "head_box_32",
        },
    )


class DetectionBindingTests(unittest.TestCase):
    def test_sdk_enum_names_are_normalized_at_the_boundary(self):
        class SdkDType:
            def __init__(self, name):
                self.name = name

        self.assertEqual(normalize_dtype(SdkDType("NV12")), np.dtype(np.uint8))
        self.assertEqual(normalize_dtype(SdkDType("U8")), np.dtype(np.uint8))
        self.assertEqual(normalize_dtype(SdkDType("F32")), np.dtype(np.float32))
        self.assertEqual(normalize_dtype(np.uint8), np.dtype(np.uint8))
        self.assertEqual(normalize_dtype(np.float32), np.dtype(np.float32))

    def test_semantic_outputs_reject_integer_and_nonfinite_values(self):
        contract = DFLDetectionContract(
            classes=1, reg_bins=16, strides=(8, 16, 32), nms="none"
        )
        outputs = {}
        for stride in contract.strides:
            height, width = 64 // stride, 96 // stride
            outputs[f"cls_{stride}"] = np.zeros(
                (1, height, width, 1), dtype=np.float32)
            outputs[f"box_{stride}"] = np.zeros(
                (1, height, width, 64), dtype=np.float32)
        with self.assertRaises(DecodeError):
            decode_dfl(
                {key: value.astype(np.int8) for key, value in outputs.items()},
                contract,
                input_size=(64, 96),
            )
        outputs["cls_8"][0, 0, 0, 0] = np.nan
        with self.assertRaises(DecodeError):
            decode_dfl(outputs, contract, input_size=(64, 96))

        nchw = {
            key: value.transpose(0, 3, 1, 2)
            for key, value in outputs.items()
            if key != "cls_8"
        }
        nchw["cls_8"] = np.zeros((1, 1, 8, 12), dtype=np.float32)
        with self.assertRaises(DecodeError):
            decode_dfl(nchw, contract, input_size=(64, 96))

        binding = bind_model(
            ModelSelection("fixture.hbm", target="x5", contract=_contract()),
            _metadata(),
        )
        physical = {
            "detector": {
                name: np.zeros(shape, dtype=np.float32)
                for name, shape in _metadata().output_shapes.items()
            }
        }
        physical["detector"]["head_cls_8"][0, 0, 0, 0] = np.inf
        with self.assertRaises(BindingError):
            binding.read_outputs(physical)

    def test_explicit_roles_bind_and_validate_rectangular_grids(self):
        selection = ModelSelection(
            model_path="fixture.hbm",
            target="x5",
            task="detect",
            contract=_contract(),
            input_shape=(64, 96),
        )

        binding = bind_model(selection, _metadata())

        self.assertEqual(binding.input_adapter.input_height, 64)
        self.assertEqual(binding.input_adapter.input_width, 96)
        self.assertEqual(binding.output_name("box", 16), "head_box_16")
        self.assertEqual(binding.grid_shape(8), (8, 12))

    def test_wrong_shape_and_integer_without_quantization_are_rejected(self):
        selection = ModelSelection(
            model_path="fixture.hbm", target="x5", task="detect", contract=_contract()
        )
        bad_shape = _metadata(output_shapes={
            "head_cls_8": (1, 8, 12, 1),
            "head_box_8": (1, 8, 11, 64),
            "head_cls_16": (1, 4, 6, 1),
            "head_box_16": (1, 4, 6, 64),
            "head_cls_32": (1, 2, 3, 1),
            "head_box_32": (1, 2, 3, 64),
        })
        with self.assertRaises(BindingError):
            bind_model(selection, bad_shape)

        with self.assertRaises(BindingError):
            bind_model(selection, _metadata(output_dtype=np.int8))

        flat_shape = {
            "head_cls_8": (1, 96, 1),
            "head_box_8": (1, 8, 12, 64),
            "head_cls_16": (1, 4, 6, 1),
            "head_box_16": (1, 4, 6, 64),
            "head_cls_32": (1, 2, 3, 1),
            "head_box_32": (1, 2, 3, 64),
        }
        with self.assertRaises(BindingError):
            bind_model(selection, _metadata(output_shapes=flat_shape))

    def test_input_binding_preserves_named_nv12_roles(self):
        adapter = InputBinding(
            model_name="detector",
            roles={"image": "nv12"},
            shapes={"nv12": (1, 3, 64, 96)},
            dtypes={"nv12": np.dtype(np.uint8)},
            packed=True,
            height=64,
            width=96,
        )
        y = np.zeros((64, 96, 1), dtype=np.uint8)
        uv = np.zeros((32, 48, 2), dtype=np.uint8)
        result = adapter.build(y, uv)
        self.assertEqual(list(result), ["detector"])
        self.assertEqual(result["detector"]["nv12"].size, 64 * 96 * 3 // 2)

    def test_letterbox_inverse_uses_actual_rounded_resize(self):
        image = np.zeros((3, 7, 3), dtype=np.uint8)
        resized, transform = resize_with_transform(image, (10, 10), resize_type=1)

        self.assertEqual(resized.shape[:2], (10, 10))
        self.assertEqual(transform.resized_size, (4, 10))
        self.assertEqual(transform.padding, (0, 3, 0, 3))
        np.testing.assert_allclose(
            inverse_boxes(np.array([[0, 3, 10, 7]], dtype=np.float32), transform),
            np.array([[0, 0, 7, 3]], dtype=np.float32),
        )

    def test_injected_runner_executes_once_and_keeps_tuple_result(self):
        contract = DFLDetectionContract(
            classes=1, reg_bins=16, strides=(8, 16, 32), nms="classwise"
        )
        adapter = InputBinding(
            model_name="fixture",
            roles={"image": "nv12"},
            shapes={"nv12": (1, 3, 64, 96)},
            dtypes={"nv12": np.dtype(np.uint8)},
            packed=True,
            height=64,
            width=96,
        )

        class FakeRunner:
            input_size = (64, 96)
            input_adapter = adapter

            def __init__(self):
                self.calls = 0

            def __call__(self, tensors):
                self.calls += 1
                self.assert_input(tensors)
                outputs = {}
                for stride in (8, 16, 32):
                    height, width = 64 // stride, 96 // stride
                    cls = np.full((1, height, width, 1), -20, dtype=np.float32)
                    cls[0, 0, 0, 0] = 10
                    box = np.zeros((1, height, width, 64), dtype=np.float32)
                    for side in range(4):
                        box[0, 0, 0, side * 16 + 1] = 10
                    outputs[f"cls_{stride}"] = cls
                    outputs[f"box_{stride}"] = box
                return outputs

            @staticmethod
            def assert_input(tensors):
                assert tensors["fixture"]["nv12"].size == 64 * 96 * 3 // 2

        fake = FakeRunner()
        model = YoloDetect(
            YoloDetectConfig(
                model_path="fixture.hbm",
                input_shape=(64, 96),
                classes_num=1,
                reg=16,
                strides=[8, 16, 32],
                score_thres=0.5,
                nms_thres=0.5,
                contract=contract,
            ),
            runner=fake,
        )
        result = model.predict(np.zeros((3, 7, 3), dtype=np.uint8))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertEqual(fake.calls, 1)
        self.assertEqual(result.boxes_xyxy.shape[1], 4)
        self.assertEqual(model.pre_process(np.zeros((3, 7, 3), dtype=np.uint8)).transform.resized_size, (41, 96))
        with self.assertRaises(BindingError):
            model.set_scheduling_params(priority=0)

    def test_runner_factory_binds_observed_opaque_output_names_by_shape(self):
        class FakeModel:
            model_names = ["detector"]
            input_names = {"detector": ["images"]}
            input_shapes = {"detector": {"images": (1, 3, 64, 96)}}
            input_dtypes = {"detector": {"images": "hbDNNDataType.NV12"}}
            output_names = {"detector": ["output0", "326", "334", "342", "350", "358"]}
            output_shapes = {"detector": {
                "output0": (1, 8, 12, 1), "326": (1, 8, 12, 64),
                "334": (1, 4, 6, 1), "342": (1, 4, 6, 64),
                "350": (1, 2, 3, 1), "358": (1, 2, 3, 64),
            }}
            output_dtypes = {"detector": {name: "hbDNNDataType.F32"
                                            for name in output_names["detector"]}}

            def run(self, tensors):
                return {"detector": {
                    name: np.zeros(shape, dtype=np.float32)
                    for name, shape in self.output_shapes["detector"].items()
                }}

        fake_model = FakeModel()
        runtime = type("Runtime", (), {
            "HB_HBMRuntime": staticmethod(lambda path: fake_model)
        })
        selection = ModelSelection(
            model_path="fixture.bin", target="x5", task="detect",
            contract=DFLDetectionContract(classes=1, reg_bins=16, strides=(8, 16, 32)),
        )
        runner = build_runner(selection, runtime_loader=lambda: runtime)
        self.assertEqual(runner.binding.output_name("cls", 8), "output0")
        semantic = runner({"detector": {"images": np.zeros(1, np.uint8)}})
        self.assertIn("box_32", semantic)


if __name__ == "__main__":
    unittest.main()
