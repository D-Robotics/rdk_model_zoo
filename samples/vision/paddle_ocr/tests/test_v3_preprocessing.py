"""Independent colored-pixel and padding checks for the converted PP-OCRv3 pair."""
from dataclasses import replace
import unittest
import numpy as np
from samples.vision.paddle_ocr.runtime.python.cli import resolve_pair
from samples.vision.paddle_ocr.runtime.python.ocr import prepare_detection, prepare_recognition, OCRPipeline

class V3PreprocessingTests(unittest.TestCase):
    def pair(self, target='s600'):
        old = resolve_pair('x5')
        det = replace(old.detector, target=target, input_protocol='bgr_imagenet_f32_nchw',
                      input_shapes={'x': (1,3,640,640)}, input_dtypes={'x':'float32'},
                      runtime_input_shapes={'x': (1,3,640,640)}, runtime_input_dtypes={'x':'float32'})
        rec = replace(old.recognizer, target=target, input_protocol='bgr_norm_pad_f32_nchw',
                      output_shape=(1,40,97))
        return replace(old, target=target, detector=det, recognizer=rec)

    def test_colored_bgr_input_and_right_padding_are_preserved(self):
        image = np.full((48,96,3), [0,127,255], dtype=np.uint8)
        pair = self.pair()
        det = prepare_detection(image, pair)['x']
        np.testing.assert_allclose(det[0,:,0,0], [-.485/.229,(127/255-.456)/.224,(1-.406)/.225], rtol=1e-6)
        rec = prepare_recognition(image, pair)['x']
        np.testing.assert_allclose(rec[0,:,0,0], [-1,127/127.5-1,1], atol=1e-7)
        self.assertEqual(rec.shape, (1,3,48,320))
        np.testing.assert_array_equal(rec[:,:,:,96:], 0)

    def test_new_s_targets_use_v3_crop_policy_and_97_class_decoder(self):
        image = np.zeros((100,200,3), np.uint8)
        for target in ('s100','s100p','s600'):
            pair = self.pair(target)
            pipeline = OCRPipeline(pair, lambda _: {}, lambda _: {})
            result = pipeline.postprocess_detection({pair.detector.output_name: np.zeros((1,1,640,640), np.float32)}, image)
            self.assertEqual(result.boxes, ())
            scores = np.zeros((1,40,97), np.float32)
            scores[:,:,0] = 1
            self.assertEqual(pipeline.decode_recognition({pair.recognizer.output_name:scores}), '')
