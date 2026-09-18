English | [简体中文](./README_cn.md)

# PaddleOCR detection and recognition

This sample runs a complete two-stage OCR flow: a DB detector finds text
regions, then a CRNN+CTC recognizer decodes each crop. The maintained Python
runtime has two audited model pairs. Choose the pair that matches the board and
keep its detector, recognizer, and dictionary together:

| Board | Model pair | Detector input | Recognizer output | Published references |
| --- | --- | --- | --- | --- |
| RDK X5 | PP-OCRv3 English | packed NV12 `[1,960,640,1]` U8 | F32 `[1,40,97,1]`, fixed 96-character alphabet plus blank | `x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin` + `x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin` |
| RDK S100 | PP-OCRv6 | split `x_y [1,640,640,1]` and `x_uv [1,320,320,2]` U8 | F32 `[1,40,18710]`, checked-in UTF-8 dictionary plus blank/space | `s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` + `s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` |

The strings in the last column qualify rows already present in the X5 and S
[release manifests](../../../platforms/x5/docs/release/models.yaml) and
[release manifest](../../../platforms/s/docs/release/models.yaml). This sample
does not infer support for S100P or S600 because no matching audited OCR pair is
published there.

## Prepare the environment and models

For Python board inference, use the RDK image that supplies the matching
`hbm_runtime` package, Python 3.10 or newer, NumPy, OpenCV-Python, PyYAML, and
`pyclipper`. `hbm_runtime` is imported only for execution; `--help`,
`--list-models`, and `--dry-run` work without the board SDK. The model files
must already exist, and the two paths must be associated with their exact
qualified references.

Model preparation is explicit. It uses the URLs in the existing release
manifest and prints the observed SHA-256; normal inference never downloads:

```bash
python samples/vision/paddle_ocr/runtime/python/main.py --prepare \
  --target x5 \
  --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
  --model-dir /tmp/rdk-models
```

Use `--target s100` with the two S100 references from the table for the S100
pair. The conversion workflow for producing new artifacts is documented in
[`conversion/README.md`](./conversion/README.md); it keeps the X5 PP-OCRv3
`hb_mapper` recipe separate from the S100 PP-OCRv6 `hb_compile` recipe.

## Run the Python sample

Inspect the available pairs before running on a board:

```bash
python samples/vision/paddle_ocr/runtime/python/main.py --list-models --target auto
python samples/vision/paddle_ocr/runtime/python/main.py --dry-run --target x5
```

After preparing X5 artifacts, a complete run is:

```bash
python samples/vision/paddle_ocr/runtime/python/main.py \
  --target x5 \
  --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
  --det-model-path /tmp/rdk-models/en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-model-path /tmp/rdk-models/en_PP-OCRv3_rec_48x320_rgb.bin \
  --test-img samples/vision/paddle_ocr/test_data/x5/paddleocr_test.jpg \
  --output-format json \
  --json-output /tmp/paddleocr-x5.json
```

For S100, replace the target, references, model paths, and fixture as follows:

```bash
python samples/vision/paddle_ocr/runtime/python/main.py \
  --target s100 \
  --det-asset-id s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec-asset-id s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --det-model-path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec-model-path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --test-img samples/vision/paddle_ocr/test_data/s100/gt_2322.jpg \
  --output-format json
```

`--priority` defaults to `0` and `--bpu-cores` to `0`; pass both when the board
application needs different scheduling. `--vocabulary-path` can point to a
replacement S100 dictionary only when its bytes match the audited dictionary
digest. The entrypoint rejects incomplete paths, mixed target references,
metadata mismatches, non-F32 outputs, and a board target that cannot be
identified.

## Read the result

Text output prints each recognized string and its ordered polygon box. JSON
contains `target`, `image_shape`, detector/recognizer asset references, and
aligned `boxes`/`texts`, for example:

```json
{
  "target": "x5",
  "boxes": [[[20, 30], [180, 30], [180, 70], [20, 70]]],
  "texts": ["RDK"]
}
```

Boxes and texts retain detector order. Returned arrays are owned by the result,
and an empty detector output skips recognition. The detector output is treated
as the observed score map and thresholded directly; no unverified activation
or accuracy claim is added. X5 detection resize is linear, S100 detection
resize is area, and both mask/recognizer resizes are linear. Contour expansion,
minimum-area filtering, crop rotation, and degenerate-crop behavior stay
target-specific.

The native S-series executable writes a side-by-side JPEG and retains the
original S16/F32 detector-output handling and FreeType text rendering. Build
and run it with:

```bash
bash samples/vision/paddle_ocr/runtime/cpp/run.sh
```

Its full dependency, model, argument, C++ API, and troubleshooting guide is
[`runtime/cpp/README.md`](./runtime/cpp/README.md). The C++ default path keeps
the historical SOC behavior; new integration evidence covers S100.

## Use the Python library

The package modules are importable from a full checkout and do not modify
`sys.path`. The entrypoint alone adds the checkout root for direct script use.
Use `resolve_pair` to bind exact assets, then compose the two lazy runners:

```python
import cv2

from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
from samples.vision.paddle_ocr.runtime.python.model_runner import create_stage_runners
from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline

pair = resolve_pair(
    "x5",
    det_asset_id="x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin",
    rec_asset_id="x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin",
    det_model_path="/tmp/rdk-models/en_PP-OCRv3_det_640x640_nv12.bin",
    rec_model_path="/tmp/rdk-models/en_PP-OCRv3_rec_48x320_rgb.bin",
)
detector, recognizer = create_stage_runners(pair, priority=0, bpu_cores=[0])
image = cv2.imread("samples/vision/paddle_ocr/test_data/x5/paddleocr_test.jpg")
result = OCRPipeline(pair, detector, recognizer).predict(image)
print(result.texts)
```

The key library modules are:

* `model_binding.py` holds the finite target contracts, manifest references,
  dictionary identity, and runtime metadata checks.
* `tensor_io.py` prepares target-specific packed/split NV12 and common RGB
  float32 NCHW recognition input.
* `model_runner.py` lazily loads one stage, validates physical tensors, and
  applies scheduling parameters.
* `geometry.py`, `decode.py`, and `pipeline.py` perform post-processing,
  target-local geometry, CTC best-path decoding, cropping, and ordered result
  ownership.

## Evaluate labeled records

Save one prediction record per image, add an `image` identifier, and compare it
with aligned ground truth using [`evaluator/README.md`](./evaluator/README.md):

```bash
python samples/vision/paddle_ocr/evaluator/evaluate.py \
  --ground-truth /data/labels.jsonl \
  --predictions /data/predictions.jsonl \
  --iou-threshold 0.5 \
  --output /data/paddleocr-evaluation.json
```

The evaluator matches boxes deterministically by IoU and reports detection
counts plus recognition agreement on matched regions. An empty GT record is
valid; recognition is marked `not_run` when there are no matched regions. A
complete labeled corpus and target-specific run are required before reporting
dataset accuracy or performance.

## Troubleshooting

* **No audited pair / mixed references:** run `--list-models`; keep the two
  references from one row and use the corresponding dictionary.
* **Model file not found:** use `--prepare` or copy an existing artifact, then
  pass both paths explicitly. The inference path never fetches a missing file.
* **Runtime metadata mismatch:** inspect the artifact with the toolchain's
  model-info command. X5 requires packed NV12 and 97 classes; S100 requires
  split NV12 and 18,710 classes.
* **`pyclipper` missing:** install it in the board Python environment before
  processing a non-empty detector result. Help/list/dry-run do not need it.
* **No boxes:** verify the image and detector artifact, then try the observed
  `0.5` threshold. A threshold change is a runtime choice, not a new accuracy
  result.
* **Garbled text:** use the dictionary paired with the recognizer. S100's
  blank/line/space order must remain unchanged; it cannot use the X5 alphabet.

## Source mapping and validation

The old Python entrypoints remain usable compatibility shims and now forward
to this canonical implementation. The old S C++ path forwards its header,
translation units, and launcher here as well. The maintained source mapping is:

| Legacy entry | Canonical implementation | Preserved contract |
| --- | --- | --- |
| X5 `PaddleOCR.pre_process` | `tensor_io.prepare_detection` | packed NV12, linear resize |
| S `PaddleOCRDet.pre_process` | `tensor_io.prepare_detection` | split NV12, area resize |
| X5/S recognition preprocessing | `tensor_io.prepare_recognition` | RGB F32 NCHW operation order |
| Legacy forward methods | `RuntimeStageRunner` | one lazy, metadata-bound runtime stage |
| Detector dilation/crop helpers | `geometry.py` + `pipeline.py` | target-local filtering/order/rotation |
| Legacy CTC converters | `decode.py` | blank reset, repeat collapse, dictionary order |
| Legacy CLIs | `runtime/python/main.py` | explicit assets, JSON output, no implicit fetch |

Host checks:

```bash
python -m unittest discover -s samples/vision/paddle_ocr/tests -p 'test_*.py' -v
```

The current evidence includes exact Python stage/input/output/lifetime checks
on X5 8GB, X5 4GB, and S100, including the legacy Python wrappers. Accuracy
and performance remain unmeasured until a labeled dataset and target-specific
measurement are supplied.
