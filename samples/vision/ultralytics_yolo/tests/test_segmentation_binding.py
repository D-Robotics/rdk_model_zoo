"""Raw segmentation transport, quantization and per-image stage contracts."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    DFLSegmentationContract,
    ModelSelection,
    RuntimeMetadata,
    bind_model,
)
from samples.vision.ultralytics_yolo.runtime.python.model_runner import ModelRunner
from samples.vision.ultralytics_yolo.runtime.python.yolo_seg import (
    YoloSeg,
    YoloSegConfig,
)


def fixture(*, chw=False, quantized=True, target="s100"):
    shapes = {
        f"{kind}_{stride}": (1, 64 // stride, 64 // stride, channels)
        for stride in (8, 16, 32)
        for kind, channels in [("cls", 1), ("box", 64), ("mces", 32)]
    }
    shapes["protos"] = (1, 32, 16, 16) if chw else (1, 16, 16, 32)
    raw = {
        name: np.full(shape, -24 if name.startswith(("cls", "box")) else 12, np.int8)
        for name, shape in shapes.items()
    }
    raw["cls_8"][0, 3, 3, 0] = 20
    for stride in (8, 16, 32):
        raw[f"box_{stride}"][..., [1, 17, 33, 49]] = 20
    proto = np.full((1, 16, 16, 32), 12, np.int8)
    proto[:, :, :7, :] = -12
    raw["protos"] = proto.transpose(0, 3, 1, 2).copy() if chw else proto
    quants = {
        name: SimpleNamespace(
            quant_type=SimpleNamespace(name="SCALE"),
            scale=np.linspace(
                0.125,
                0.375,
                shape[1] if chw and name == "protos" else shape[-1],
                dtype=np.float32,
            ),
            zero_point=np.array([3], np.int32),
            axis=1 if chw and name == "protos" else 3,
        )
        for name, shape in shapes.items()
    }
    if not quantized:
        raw = {
            name: (value.astype(np.float32) - 3)
            * quants[name].scale.reshape(
                (1, -1, 1, 1) if chw and name == "protos" else (1, 1, 1, -1)
            )
            for name, value in raw.items()
        }
        quants = {}
    ins = (
        {"image": (1, 3, 64, 64)}
        if target == "x5"
        else {"y": (1, 64, 64, 1), "uv": (1, 32, 32, 2)}
    )
    runtime = SimpleNamespace(
        model_names=["m"],
        input_names={"m": list(ins)},
        input_shapes={"m": ins},
        input_dtypes={"m": {n: "uint8" for n in ins}},
        # Deliberately reverse runtime enumeration. Shape roles must win.
        output_names={"m": list(reversed(shapes))},
        output_shapes={"m": shapes},
        output_dtypes={"m": {n: a.dtype for n, a in raw.items()}},
        output_quants={"m": quants},
        run=lambda inputs: {"m": raw},
    )
    contract = DFLSegmentationContract(classes=1)
    metadata = RuntimeMetadata.from_runtime(runtime)
    binding = bind_model(
        ModelSelection("fixture.hbm", target=target, task="segment", contract=contract),
        metadata,
    )
    runner = ModelRunner(runtime, binding, metadata)
    task = YoloSeg(
        YoloSegConfig("fixture.hbm", classes_num=1, nms_thres=0.7), runner=runner
    )
    return task, raw, quants


class SegmentationBinding(unittest.TestCase):
    def assert_result_equal(self, actual, expected):
        for a, b in zip(actual[:3], expected[:3]):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-5)
        self.assertEqual(len(actual[3]), len(expected[3]))
        for a, b in zip(actual[3], expected[3]):
            np.testing.assert_array_equal(a, b)

    def test_quantized_raw_forward_and_float_decode_agree_both_proto_layouts(self):
        for target in ("x5", "s100", "s600"):
            for chw in (False, True):
                with self.subTest(target=target, chw=chw):
                    task, raw, _ = fixture(chw=chw, target=target)
                    reference, _, _ = fixture(chw=chw, quantized=False, target=target)
                    outputs = task.forward({})
                    for name, array in raw.items():
                        self.assertIs(outputs[name], array)
                        self.assertEqual(array.dtype, np.int8)
                    actual = task.post_process(outputs, 64, 64)
                    self.assertEqual(len(actual[0]), 1)
                    self.assert_result_equal(
                        actual, reference.post_process(reference.forward({}), 64, 64)
                    )

    def test_border_masks_clip_to_visible_content_instead_of_negative_slices(self):
        task, raw, _ = fixture(quantized=False)
        raw["protos"].fill(1)
        raw["cls_8"].fill(-20)
        raw["cls_8"][0, 0, 0, 0] = 8
        raw["box_8"].fill(-20)
        raw["box_8"][..., [4, 20, 36, 52]] = 20
        actual = task.post_process(task.forward({}), 64, 64)
        np.testing.assert_array_equal(actual[0][0], [0, 0, 36, 36])
        self.assertTrue(
            np.all(actual[3][0] == 1),
            "Visible mask must not be lost by negative NumPy slicing",
        )
        # A small box lies wholly in top letterbox padding of a panoramic image.
        raw["box_8"].fill(-20)
        raw["box_8"][..., [0, 16, 32, 48]] = 20
        prepared = task.pre_process(np.zeros((8, 64, 3), np.uint8))
        result = task.post_process(task.forward(prepared), transform=prepared.transform)
        self.assertEqual(result[0][0, 1], result[0][0, 3])
        self.assertFalse(np.any(result[3][0]))

    def test_empty_and_owned_results(self):
        task, raw, _ = fixture()
        result = task.post_process(task.forward({}), 64, 64)
        saved = [array.copy() for array in (*result[:3], *result[3])]
        for array in raw.values():
            array.fill(-24)
        for actual, expected in zip((*result[:3], *result[3]), saved):
            np.testing.assert_array_equal(actual, expected)
        empty = task.post_process(task.forward({}), 64, 64)
        self.assertEqual(empty[0].shape, (0, 4))
        self.assertEqual(empty[0].dtype, np.float32)
        self.assertEqual(empty[1].shape, (0,))
        self.assertEqual(empty[2].dtype, np.int64)
        self.assertEqual(empty[3], [])

    def test_interleaved_context_and_predict_agree(self):
        task, _, _ = fixture()
        for resize in (0, 1):
            task.cfg.resize_type = resize
            a, b = np.zeros((31, 73, 3), np.uint8), np.zeros((67, 23, 3), np.uint8)
            pa, pb = task.pre_process(a), task.pre_process(b)
            self.assertEqual(pa.transform.original_size, (31, 73))
            self.assertEqual(pb.transform.original_size, (67, 23))
            self.assert_result_equal(
                task.post_process(task.forward(pa), transform=pa.transform),
                task.predict(a),
            )
            self.assert_result_equal(
                task.post_process(task.forward(pb), transform=pb.transform),
                task.predict(b),
            )
            with self.assertRaises(ValueError):
                task.post_process(task.forward({}))
            with self.assertRaises(ValueError):
                task.post_process(task.forward({}), 999, 999, transform=pa.transform)

    def test_metadata_failures_are_rejected_before_execution(self):
        task, _, _ = fixture()
        binding = task.binding
        metadata = binding.metadata
        for changes in [
            {"output_quantization": {}},
            {"output_shapes": {**metadata.output_shapes, "protos": (1, 15, 16, 32)}},
            {"output_shapes": {**metadata.output_shapes, "mces_8": (1, 8, 8, 31)}},
        ]:
            with self.subTest(changes=changes.keys()), self.assertRaises(ValueError):
                bind_model(binding.selection, replace(metadata, **changes))
        other, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, "different"):
            task.post_process(other.forward({}), 64, 64)

    def test_ambiguous_roles_require_reviewed_names(self):
        task, _, _ = fixture(quantized=False)
        metadata = task.binding.metadata
        shapes = dict(metadata.output_shapes)
        for stride in (8, 16, 32):
            shapes[f"cls_{stride}"] = (1, 64 // stride, 64 // stride, 32)
        metadata = replace(metadata, output_shapes=shapes)
        contract = DFLSegmentationContract(classes=32)
        selection = replace(task.binding.selection, contract=contract)
        with self.assertRaisesRegex(ValueError, "Multiple tensors"):
            bind_model(selection, metadata)
        reviewed = DFLSegmentationContract(
            classes=32, output_roles={name: name for name in contract.required_roles}
        )
        binding = bind_model(replace(selection, contract=reviewed), metadata)
        self.assertEqual(binding.output_adapter.role_to_name["cls_8"], "cls_8")
        self.assertEqual(binding.output_adapter.role_to_name["mces_8"], "mces_8")

    def test_bad_semantic_input_and_thresholds_fail_explicitly(self):
        task, raw, _ = fixture()
        with self.assertRaises(ValueError):
            task.post_process(raw, 64, 64)
        for value in (0, 1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                task.post_process(task.forward({}), 64, 64, score_thres=value)
        for image in (np.zeros((0, 2, 3), np.uint8), np.zeros((2, 2, 3), np.float32)):
            with self.assertRaises(ValueError):
                task.pre_process(image)


if __name__ == "__main__":
    unittest.main()
