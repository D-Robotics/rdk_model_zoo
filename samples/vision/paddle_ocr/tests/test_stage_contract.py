"""Stage-contract tests for OCRPipeline (inference-contract §3/§5)."""

from __future__ import annotations

import unittest

import numpy as np


def _detector_mask(value: float = 0.0) -> np.ndarray:
    mask = np.full((1, 1, 640, 640), value, dtype=np.float32)
    return mask


def _rec_scores() -> np.ndarray:
    scores = np.full((1, 40, 97, 1), -3.0, dtype=np.float32)
    scores[:, :, 1, :] = 3.0
    return scores


class StageContractTests(unittest.TestCase):
    def _pipeline(self, detector, recognizer):
        from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
        from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline

        return OCRPipeline(resolve_pair("x5"), detector, recognizer)

    def test_predict_equals_explicit_stage_composition(self):
        from samples.vision.paddle_ocr.runtime.python.pipeline import DetectionResult

        box = np.array([[1, 1], [10, 1], [10, 5], [1, 5]], dtype=np.int64)
        crop = np.full((7, 11, 3), 23, dtype=np.uint8)
        pipeline = self._pipeline(
            lambda inputs: {"sigmoid_0.tmp_0": _detector_mask()},
            lambda inputs: {"softmax_2.tmp_0": _rec_scores()},
        )
        pipeline.postprocess_detection = lambda outputs, image: DetectionResult(
            boxes=(box,), crops=(crop,)
        )
        image = np.zeros((32, 48, 3), dtype=np.uint8)

        via_predict = pipeline.predict(image)

        detection = pipeline.run_detection(image)
        det_outputs = pipeline.forward_detection(pipeline.prepare_detection(image))
        self.assertEqual(
            len(detection.boxes), len(pipeline.postprocess_detection(det_outputs, image).boxes))
        texts = tuple(
            pipeline.decode_recognition(
                pipeline.forward_recognition(pipeline.prepare_recognition(one_crop)))
            for one_crop in detection.crops
        )

        self.assertEqual(via_predict.texts, texts)
        self.assertEqual(via_predict.boxes[0].tolist(), detection.boxes[0].tolist())

    def test_forward_detection_returns_validated_raw_without_decoding(self):
        raw = _detector_mask(0.25)
        raw[0, 0, 7, 11] = 0.9
        pipeline = self._pipeline(
            lambda inputs: {"sigmoid_0.tmp_0": raw},
            lambda inputs: {"softmax_2.tmp_0": _rec_scores()},
        )
        inputs = pipeline.prepare_detection(np.zeros((32, 48, 3), dtype=np.uint8))

        outputs = pipeline.forward_detection(inputs)

        # Container adaptation only: bit-identical raw tensor, no thresholding
        # to 0/255, no contour extraction, no file access.
        np.testing.assert_array_equal(outputs["sigmoid_0.tmp_0"], raw)

    def test_forward_recognition_returns_validated_raw_without_decoding(self):
        raw = _rec_scores()
        pipeline = self._pipeline(
            lambda inputs: {"sigmoid_0.tmp_0": _detector_mask()},
            lambda inputs: {"softmax_2.tmp_0": raw},
        )
        inputs = pipeline.prepare_recognition(np.full((7, 11, 3), 23, dtype=np.uint8))

        outputs = pipeline.forward_recognition(inputs)

        # No CTC decoding, no argmax collapse: raw scores pass through intact.
        np.testing.assert_array_equal(outputs["softmax_2.tmp_0"], raw)

    def test_public_forward_attributes_errors_to_their_stage(self):
        def failing_detector(inputs):
            raise KeyError("det-boom")

        def failing_recognizer(inputs):
            raise KeyError("rec-boom")

        det_pipeline = self._pipeline(failing_detector, lambda inputs: {})
        with self.assertRaises(RuntimeError) as ctx:
            det_pipeline.forward_detection(
                det_pipeline.prepare_detection(np.zeros((32, 48, 3), dtype=np.uint8)))
        self.assertIn("detector", str(ctx.exception))

        rec_pipeline = self._pipeline(lambda inputs: {}, failing_recognizer)
        with self.assertRaises(RuntimeError) as ctx:
            rec_pipeline.forward_recognition(
                rec_pipeline.prepare_recognition(np.full((7, 11, 3), 9, dtype=np.uint8)))
        self.assertIn("recognizer", str(ctx.exception))

    def test_recognizer_failure_keeps_crop_attribution(self):
        from samples.vision.paddle_ocr.runtime.python.pipeline import DetectionResult

        calls = []

        def recognizer(inputs):
            calls.append(inputs)
            if len(calls) == 2:
                raise ValueError("boom-on-second-crop")
            return {"softmax_2.tmp_0": _rec_scores()}

        pipeline = self._pipeline(
            lambda inputs: {"sigmoid_0.tmp_0": _detector_mask()},
            recognizer,
        )
        pipeline.postprocess_detection = lambda outputs, image: DetectionResult(
            boxes=(np.array([[1, 1]], dtype=np.int64),
                   np.array([[2, 2]], dtype=np.int64)),
            crops=(np.full((4, 4, 3), 1, dtype=np.uint8),
                   np.full((4, 4, 3), 2, dtype=np.uint8)),
        )

        with self.assertRaises(RuntimeError) as ctx:
            pipeline.predict(np.zeros((32, 48, 3), dtype=np.uint8))

        message = str(ctx.exception)
        self.assertIn("recognizer stage failed for crop 1", message)
        self.assertIn("boom-on-second-crop", message)


if __name__ == "__main__":
    unittest.main()
