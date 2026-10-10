"""NMS-free v10 must share raw stages without changing the X5 dispatch policy."""

from dataclasses import replace
import unittest
import numpy as np
from test_forward_purity import fixture as detection_fixture
from samples.vision.ultralytics_yolo.runtime.python.backend import (
    DFLDetectionContract,
    ModelSelection,
    bind_model,
)
from samples.vision.ultralytics_yolo.runtime.python.backend import ModelRunner
from samples.vision.ultralytics_yolo.runtime.python.detect import (
    YoloV10Detect,
    YoloV10DetectConfig,
)
from samples.vision.ultralytics_yolo.runtime.python.detect import YoloDetect


def fixture():
    previous, physical, _, _ = detection_fixture()
    contract = DFLDetectionContract(classes=1, nms="none")
    metadata = replace(
        previous.metadata,
        input_names=("y", "uv"),
        input_shapes={"y": (1, 64, 64, 1), "uv": (1, 32, 32, 2)},
        input_dtypes={"y": np.uint8, "uv": np.uint8},
    )
    binding = bind_model(
        ModelSelection("stub", target="s100", contract=contract), metadata
    )
    model = YoloV10Detect(
        YoloV10DetectConfig("stub", classes_num=1),
        runner=ModelRunner(previous.model, binding),
    )
    return model, physical


class V10Stages(unittest.TestCase):
    def test_raw_identity_overlap_and_source_order(self):
        model, raw = fixture()
        raw["cls_8"][0, 3, 3, 0] = 2
        raw["cls_8"][0, 3, 4, 0] = 4
        for stride in (8, 16, 32):
            raw[f"box_{stride}"].fill(-24)
            raw[f"box_{stride}"][..., [3, 19, 35, 51]] = 16
        output = model.forward({})
        self.assertIs(output["box_8"], raw["box_8"])
        result = model.post_process(output, 64, 64)
        self.assertEqual(len(result.scores), 2)
        self.assertLess(result.scores[0], result.scores[1])
        saved = result.boxes.copy()
        raw["box_8"].fill(0)
        np.testing.assert_array_equal(result.boxes, saved)
        self.assertIs(YoloV10Detect.pre_process, YoloDetect.pre_process)
        self.assertIs(YoloV10Detect.forward, YoloDetect.forward)
        self.assertIs(YoloV10Detect.post_process, YoloDetect.post_process)

    def test_context_interleaving_empty_and_predict(self):
        model, raw = fixture()
        a = np.zeros((37, 59, 3), np.uint8)
        b = np.zeros((83, 17, 3), np.uint8)
        pa = model.pre_process(a)
        pb = model.pre_process(b)
        ra = model.post_process(model.forward(pa), transform=pa.transform)
        rb = model.post_process(model.forward(pb), transform=pb.transform)
        np.testing.assert_allclose(
            ra.boxes,
            [[20 * 59 / 64, 8 * 37 / 40, 36 * 59 / 64, 24 * 37 / 40]],
            rtol=1e-6,
        )
        for x, y in zip(ra, model.predict(a)):
            np.testing.assert_array_equal(x, y)
        for x, y in zip(rb, model.predict(b)):
            np.testing.assert_array_equal(x, y)
        for stride in (8, 16, 32):
            raw[f"cls_{stride}"].fill(-24)
        empty = model.predict(a)
        self.assertEqual(empty.boxes.shape, (0, 4))
        self.assertEqual(empty.scores.shape, (0,))
        self.assertEqual(empty.class_ids.dtype, np.int64)

    def test_confidence_endpoints_and_invalid_values(self):
        model, _ = fixture()
        raw = model.forward({})
        self.assertEqual(len(model.post_process(raw, 64, 64, score_thres=0).scores), 84)
        self.assertEqual(len(model.post_process(raw, 64, 64, score_thres=1).scores), 0)
        expected = model.post_process(raw, 64, 64)
        ignored = model.post_process(raw, 64, 64, nms_thres=0)
        for a, b in zip(expected, ignored):
            np.testing.assert_array_equal(a, b)
        for value in (-.1, 1.1, np.nan, np.inf):
            with self.subTest(threshold=value), self.assertRaises(ValueError):
                model.post_process(raw, 64, 64, score_thres=value)

    def test_nms_contract_cannot_be_overridden(self):
        runner, _, _, _ = detection_fixture()
        with self.assertRaises(ValueError):
            YoloV10Detect(YoloV10DetectConfig("stub"), runner=runner)
        with self.assertRaises(ValueError):
            YoloV10Detect(YoloV10DetectConfig("stub", contract=DFLDetectionContract()))


if __name__ == "__main__":
    unittest.main()
