# Test data provenance

The pilot copies only the fixtures needed for its default X5 and S100 Python
comparisons. The bytes are unchanged from the existing platform samples.

| File | Original source | SHA-256 | License/source note |
| --- | --- | --- | --- |
| `x5/paddleocr_test.jpg` | [`platforms/x5/samples/vision/paddleocr/test_data/paddleocr_test.jpg`](../../../../platforms/x5/samples/vision/paddleocr/test_data/paddleocr_test.jpg) | `5b4a7fb523c7c459c8d3cec67480c1872cd7b3674b34505467420561ad8c577e` | Existing X5 sample; see [X5 Apache-2.0 license](../../../../platforms/x5/LICENSE) |
| `s100/gt_2322.jpg` | [`platforms/s/samples/vision/paddle_ocr/test_data/gt_2322.jpg`](../../../../platforms/s/samples/vision/paddle_ocr/test_data/gt_2322.jpg) | `18a214e1c637fb3a53f71673c6f6a689b5f16d755237ab7e9e58ddc32223580b` | Existing S sample; see [S Apache-2.0 license](../../../../platforms/s/LICENSE) |
| `s100/ppocrv6_dict.txt` | [`platforms/s/samples/vision/paddle_ocr/test_data/ppocrv6_dict.txt`](../../../../platforms/s/samples/vision/paddle_ocr/test_data/ppocrv6_dict.txt) | `769e7fa79bb297b5f18d8dbd149e364a45bc61f2b3f574e5ea836f0b261c23a6` | Existing S sample dictionary; see [S Apache-2.0 license](../../../../platforms/s/LICENSE) |

The two image files are used as BGR OpenCV inputs. The S100 dictionary is
decoded as UTF-8 with one source line per token, then the pilot adds the blank
class at index zero and the source-compatible trailing space. No model files,
fonts or generated result images are copied into this candidate sample.
