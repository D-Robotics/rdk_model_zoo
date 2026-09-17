"""Host tests for OCR decoding and the injected two-stage pipeline seam."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np


class DecodePipelineTests(unittest.TestCase):
    def test_ctc_blank_resets_repeat_and_unicode_tokens_are_preserved(self):
        from samples.vision.paddle_ocr.runtime.python.decode import ctc_greedy_decode

        tokens = ("blank", "你", "好", " ")
        scores = np.full((1, 5, len(tokens)), -10.0, dtype=np.float32)
        for time, index in enumerate((1, 1, 0, 1, 2)):
            scores[0, time, index] = 10.0
        self.assertEqual(ctc_greedy_decode(scores, tokens), "你你好")

    def test_ctc_rejects_bad_class_count_and_nonfinite_scores(self):
        from samples.vision.paddle_ocr.runtime.python.decode import ctc_greedy_decode

        with self.assertRaises(ValueError):
            ctc_greedy_decode(np.zeros((1, 3, 4), dtype=np.float32), ("blank", "a"))
        bad = np.zeros((1, 3, 2), dtype=np.float32)
        bad[0, 0, 1] = np.nan
        with self.assertRaises(ValueError):
            ctc_greedy_decode(bad, ("blank", "a"))

    def test_zero_detector_output_returns_empty_owned_result_without_recognition(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
        from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline

        calls = []

        def detector(inputs):
            calls.append(("detector", inputs))
            return {"sigmoid_0.tmp_0": np.zeros((1, 1, 640, 640), dtype=np.float32)}

        def recognizer(inputs):
            calls.append(("recognizer", inputs))
            return {"softmax_2.tmp_0": np.zeros((1, 40, 97, 1), dtype=np.float32)}

        pipeline = OCRPipeline(resolve_pair("x5"), detector, recognizer)
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        result = pipeline.predict(image)
        self.assertEqual(result.boxes, ())
        self.assertEqual(result.texts, ())
        self.assertEqual([name for name, _ in calls], ["detector"])

    def test_postprocess_detection_returns_boxes_and_crops(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
        from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline

        pipeline = OCRPipeline(
            resolve_pair("x5"),
            lambda inputs: {"sigmoid_0.tmp_0": np.zeros((1, 1, 640, 640), dtype=np.float32)},
            lambda inputs: {"softmax_2.tmp_0": np.zeros((1, 40, 97, 1), dtype=np.float32)},
        )
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        detection = pipeline.postprocess_detection(
            {"sigmoid_0.tmp_0": np.zeros((1, 1, 640, 640), dtype=np.float32)}, image
        )
        self.assertEqual(detection.boxes, ())
        self.assertEqual(detection.crops, ())

    def test_stage_output_shape_dtype_and_nonfinite_values_are_rejected(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import (
            MetadataMismatchError,
            resolve_pair,
        )
        from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline

        pair = resolve_pair("x5")
        pipeline = OCRPipeline(pair, lambda inputs: {}, lambda inputs: {})
        with self.assertRaises(MetadataMismatchError):
            pipeline.postprocess_detection(
                {"sigmoid_0.tmp_0": np.zeros((1, 1, 640, 639), dtype=np.float32)},
                np.zeros((40, 60, 3), dtype=np.uint8),
            )
        bad = np.zeros((1, 1, 640, 640), dtype=np.float32)
        bad[0, 0, 0, 0] = np.inf
        with self.assertRaises(MetadataMismatchError):
            pipeline.postprocess_detection({"sigmoid_0.tmp_0": bad}, np.zeros((40, 60, 3), dtype=np.uint8))

    def test_both_injected_stages_keep_order_and_result_lifetime_across_calls(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
        from samples.vision.paddle_ocr.runtime.python.pipeline import (
            DetectionResult,
            OCRPipeline,
        )

        pair = resolve_pair("x5")
        detector_calls = []
        recognizer_calls = []
        box = np.array([[1, 1], [10, 1], [10, 5], [1, 5]], dtype=np.int64)
        crop = np.full((7, 11, 3), 23, dtype=np.uint8)

        def detector(inputs):
            detector_calls.append(inputs["x"].copy())
            return {
                "sigmoid_0.tmp_0": np.zeros((1, 1, 640, 640), dtype=np.float32)
            }

        def recognizer(inputs):
            recognizer_calls.append(inputs["x"].copy())
            scores = np.full((1, 40, 97, 1), -3, dtype=np.float32)
            scores[:, :, 1, :] = 3
            return {"softmax_2.tmp_0": scores}

        pipeline = OCRPipeline(pair, detector, recognizer)
        # Keep this test independent of optional pyclipper while exercising the
        # public stage composition and returned-array ownership.
        pipeline.postprocess_detection = lambda outputs, image: DetectionResult(
            boxes=(box,), crops=(crop,)
        )

        first = pipeline.predict(np.zeros((32, 48, 3), dtype=np.uint8))
        box[0, 0] = 99
        crop[0, 0, 0] = 99
        second = pipeline.predict(np.zeros((32, 48, 3), dtype=np.uint8))
        self.assertEqual(first.texts, ("0",))
        self.assertEqual(second.texts, ("0",))
        self.assertEqual(first.boxes[0][0, 0], 1)
        self.assertEqual(first.boxes[0].shape, (4, 2))
        self.assertEqual(first.texts, second.texts)
        self.assertEqual(len(detector_calls), 2)
        self.assertEqual(len(recognizer_calls), 2)

    def test_postprocess_exposes_owned_rotated_crop_for_a_detected_polygon(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
        from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline

        pair = resolve_pair("x5")
        pipeline = OCRPipeline(pair, lambda inputs: {}, lambda inputs: {})
        polygon = np.array(
            [[[4, 8], [54, 8], [54, 28], [4, 28]]], dtype=np.int64
        )
        output = np.zeros((1, 1, 640, 640), dtype=np.float32)
        image = np.arange(64 * 80 * 3, dtype=np.uint16).reshape(64, 80, 3)
        image = (image % 251).astype(np.uint8)
        with patch(
            "samples.vision.paddle_ocr.runtime.python.pipeline.dilate_contours",
            return_value=(polygon,),
        ):
            detection = pipeline.postprocess_detection(
                {"sigmoid_0.tmp_0": output}, image
            )
        self.assertEqual(len(detection.boxes), 1)
        self.assertEqual(len(detection.crops), 1)
        self.assertEqual(detection.boxes[0].shape, (4, 2))
        self.assertEqual(detection.crops[0].ndim, 3)
        self.assertEqual(detection.crops[0].shape[2], 3)

    def test_postprocess_keeps_dilated_polygons_filtered_from_boxes(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
        from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline

        pair = resolve_pair("s100")
        pipeline = OCRPipeline(pair, lambda inputs: {}, lambda inputs: {})
        small = np.array(
            [[[2, 2], [6, 2], [6, 5], [2, 5]]], dtype=np.int64
        )
        large = np.array(
            [[[10, 10], [50, 10], [50, 30], [10, 30]]], dtype=np.int64
        )
        with patch(
            "samples.vision.paddle_ocr.runtime.python.pipeline.dilate_contours",
            return_value=(small, large),
        ):
            detection = pipeline.postprocess_detection(
                {"fetch_name_0": np.zeros((1, 1, 640, 640), dtype=np.float32)},
                np.zeros((64, 80, 3), dtype=np.uint8),
            )
        self.assertEqual(len(detection.polygons), 2)
        self.assertEqual(len(detection.boxes), 1)
        self.assertEqual(len(detection.crops), 1)


if __name__ == "__main__":
    unittest.main()
