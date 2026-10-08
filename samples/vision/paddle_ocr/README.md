English | [简体中文](./README_cn.md)

# PaddleOCR two-stage text detection and recognition

<a id="overview"></a>
## Overview

This sample runs a complete two-stage OCR flow on RDK boards: a DB detector
finds text regions in the input image, each detected region is cropped, and a
CRNN+CTC recognizer decodes the crop into a string. It is the repository's
reference implementation for legal multi-stage inference (see
[stage I/O](./runtime/python/README.md#stage-io)): detection and recognition
are separate lazily loaded runtime stages composed by an explicit pipeline,
and the detect→crop→recognize order stays readable end to end.

Two model pairs are provided. A pair is detector + recognizer +
dictionary as a unit; never mix components across pairs:

| Board | Pair | Detector input | Recognizer output |
| --- | --- | --- | --- |
| RDK X5 | PP-OCRv3 English | one packed NV12 tensor (640×640) | F32 `[1,40,97,1]`: fixed 96-character alphabet plus blank |
| RDK S100 | PP-OCRv6 | split NV12 `x_y` (640×640) + `x_uv` (320×320) | F32 `[1,40,18710]`: checked-in UTF-8 dictionary plus blank and trailing space |

<a id="directory"></a>
## Directory structure

```text
paddle_ocr/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python and native inference implementations
├── test_data/  # Example inputs
├── tests/  # Automated tests
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── __init__.py  # Python script
```

<a id="support-matrix"></a>
## Support matrix

| Board | Python runtime | C++ runtime |
| --- | --- | --- |
| X5 | supported | not-supported (no X5 C++ implementation provided) |
| S100 | supported | supported |
| S100P | not-supported (no OCR pair published) | not-supported |
| S600 | supported | supported |

The Python default and aspect-ratio pipelines (including the command-line
helpers) run on X5 and S100 and emit the detector/recognizer stage tensors,
decoded into polygon boxes and recognized text; the S100 C++ build renders
the recognized results onto the output image. S600
shares source and SoC detection with S100; run the S600 commands on an S600
board for S600 results. S100P has no matching pair in either release
manifest, so the sample rejects it.

<a id="prerequisites"></a>
## Prerequisites

- A board image that ships the `hbm_runtime` package matching the target
  (X5 image for the `.bin` pair, RDK S image for the `.hbm` pair).
- Python 3.10 or newer with NumPy, OpenCV-Python, and PyYAML; `pyclipper`
  is additionally required whenever the detector returns at least one box
  (help, list, and dry-run modes need none of the board SDK packages).
- Prepared model artifacts (see [model preparation](#entry-points)); the
  inference path never downloads.
- All commands below run from the repository root of a full checkout.

<a id="quickstart"></a>
## Quickstart

1. List the manifest-backed pairs (no SDK needed; success: two rows printed
   with qualified references):

   ```bash
   python3 samples/vision/paddle_ocr/runtime/python/main.py \
     --list-models --target auto
   ```

2. Prepare the X5 pair explicitly — the only network-capable operation
   (inputs: manifest URLs; output: two `.bin` files under `--model-dir`;
   success: exit 0 and printed observed SHA-256 digests):

   ```bash
   python3 samples/vision/paddle_ocr/runtime/python/main.py --prepare \
     --target x5 \
     --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
     --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
     --model-dir /tmp/rdk-models
   ```

3. Run the X5 pair (cwd: repository root; success: exit 0, recognized
   strings with polygon boxes on stdout):

   ```bash
   python3 samples/vision/paddle_ocr/runtime/python/main.py \
     --target x5 \
     --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
     --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
     --det-model-path /tmp/rdk-models/en_PP-OCRv3_det_640x640_nv12.bin \
     --rec-model-path /tmp/rdk-models/en_PP-OCRv3_rec_48x320_rgb.bin \
     --test-img samples/vision/paddle_ocr/test_data/x5/paddleocr_test.jpg \
     --output-format json
   ```

4. On S100, switch target, references, paths, and fixture together (the
   pair below uses the artifacts already present under
   `/opt/hobot/model/s100/basic` on S images):

   ```bash
   python3 samples/vision/paddle_ocr/runtime/python/main.py \
     --target s100 \
     --det-asset-id s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
     --rec-asset-id s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
     --det-model-path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
     --rec-model-path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
     --test-img samples/vision/paddle_ocr/test_data/s100/gt_2322.jpg \
     --output-format json
   ```

<a id="expected-results"></a>
## Expected results

Text output prints each recognized string with its ordered polygon box.
JSON output contains `target`, `image_shape`, `detector_asset`,
`recognizer_asset`, and aligned `boxes`/`texts`, for example:

```json
{
  "target": "x5",
  "boxes": [[[20, 30], [180, 30], [180, 70], [20, 70]]],
  "texts": ["RDK"]
}
```

Boxes and texts keep detector order; arrays in the result are owned by the
result. An empty detector output skips recognition and returns empty lists.
The detector output is treated as the observed score map and thresholded
directly (`0.5`); What
the bundled fixtures print exactly is a property of the artifact pair — see
[evaluation](./evaluator/README.md#reference-results) for the verified
comparison record; dataset-level accuracy requires a labeled corpus and is not bundled.

<a id="entry-points"></a>
## Entry points

- Models: [model/README.md](./model/README.md) — the four artifact
  references and the `--prepare` procedure.
- Python runtime: [runtime/python/README.md](./runtime/python/README.md) —
  full parameter table, integration example, and stage I/O contract.
- C++ runtime: [runtime/cpp/README.md](./runtime/cpp/README.md) — build,
  run, gflags, and lifecycle for the S-series executable.
- Conversion: [conversion/README.md](./conversion/README.md) — PP-OCRv3
  `hb_mapper` and PP-OCRv6 `hb_compile` recipes.
- Evaluation: [evaluator/README.md](./evaluator/README.md) — labeled-record
  evaluation and its boundaries.

<a id="license"></a>
## License

Sample code follows the repository license. The model artifacts are
published through the platform release manifests; the PP-OCRv3 and
PP-OCRv6 model weights are PaddlePaddle upstream releases and their use is
governed by the corresponding upstream license. The S100 dictionary and the
C++ demo font are carried from the source deliveries unchanged.
