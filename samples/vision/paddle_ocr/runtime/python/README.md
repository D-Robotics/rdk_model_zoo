# PaddleOCR Python runtime

`main.py` is the canonical entrypoint for the X5 PP-OCRv3 and S100 PP-OCRv6
pairs. It resolves a qualified detector/recognizer pair, checks the execution
target against detected hardware, loads each stage lazily, and emits ordered
boxes and texts.

<a id="environment"></a>
## Environment

- Board image shipping `hbm_runtime` for the target (X5 image for `.bin`,
  RDK S image for `.hbm`); imported only at execution time.
- Python 3.10+, NumPy, OpenCV-Python, PyYAML always; `pyclipper` when the
  detector returns at least one box. `--help`, `--list-models`, and
  `--dry-run` run on a plain host without any board SDK package.
- Both model paths prepared in advance (see
  [model preparation](../../model/README.md#preparation)); inference never
  downloads.
- All commands below use cwd = repository root of a full checkout. The
  package modules use absolute full-checkout imports and do not modify
  `sys.path`; only `main.py` supports direct script invocation from an
  arbitrary directory.

<a id="usage"></a>
## Usage

Inspection modes (no SDK, no models needed; success: exit 0):

```bash
python3 samples/vision/paddle_ocr/runtime/python/main.py --help
python3 samples/vision/paddle_ocr/runtime/python/main.py --list-models --target auto
python3 samples/vision/paddle_ocr/runtime/python/main.py --dry-run --target x5
```

Complete X5 inference (cwd: repository root; success: exit 0, JSON on
stdout):

```bash
python3 samples/vision/paddle_ocr/runtime/python/main.py \
  --target x5 \
  --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
  --det-model-path /tmp/rdk-models/en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-model-path /tmp/rdk-models/en_PP-OCRv3_rec_48x320_rgb.bin \
  --test-img samples/vision/paddle_ocr/test_data/x5/paddleocr_test.jpg \
  --output-format json \
  --json-output /tmp/paddleocr-x5.json
```

Complete S100 inference (swap target, references, paths, and fixture
together, as one pair):

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

`--prepare` with `--model-dir` is the explicit model-fetch operation (see
[model preparation](../../model/README.md#preparation)). The entrypoint
rejects incomplete path pairs, mixed-target references, runtime metadata
mismatches, non-F32 outputs, and an execution target that cannot be
identified — it never guesses from a filename.

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | choice | auto | execution target: `auto`, `x5`, `s100`, `s100p`, `s600`; real execution requires an exact detected target |
| `--det-asset-id` | string | null | qualified detector reference `group:sample:filename` from `--list-models` |
| `--rec-asset-id` | string | null | qualified recognizer reference for the same pair |
| `--det-model-path` | string | null | existing local detector artifact; never downloaded implicitly; defaults to `model/<filename>` lookup when omitted |
| `--rec-model-path` | string | null | existing local recognizer artifact; required together with the detector path |
| `--vocabulary-path` | string | null | optional S100 dictionary replacement; accepted only on the audited digest |
| `--test-img` | string | null | BGR input image; when omitted, the fixture of the resolved pair is used |
| `--output-format` | choice | text | `text` or `json` result rendering |
| `--json-output` | string | null | also write the JSON inference result to this path |
| `--priority` | int | 0 | runtime scheduling priority (0-255) |
| `--bpu-cores` | int list | [0] | runtime BPU core indexes |
| `--model-dir` | string | null | destination directory for the explicit `--prepare` operation |
| `--list-models` | flag | false | list manifest-backed pairs without SDK, OpenCV, or pyclipper |
| `--dry-run` | flag | false | resolve a pair and print its static contract without model loading |
| `--prepare` | flag | false | explicitly fetch the selected manifest assets into local paths |

`--list-models`, `--dry-run`, and `--prepare` are mutually exclusive modes.
Defaults above are machine-checked against `build_parser()` by the Q3
checker.

<a id="results"></a>
## Results

Text output prints each recognized string and its ordered polygon box.
JSON contains `target`, `image_shape`, `detector_asset`, `recognizer_asset`,
and aligned `boxes`/`texts` (example in the
[sample README](../../README.md#expected-results)); `--json-output` writes
the same object to a file. Boxes and texts keep detector order; returned
arrays are owned by the result; an empty detector output skips recognition.
Model output semantics remain the observed score-map/CTC policies — no
unverified activation is inserted. `s100p` and `s600` have no audited
PaddleOCR pair and are rejected at pair resolution.

<a id="integration-example"></a>
## Integration example

Compose the two stages directly from a full checkout (imports are absolute;
all inputs defined; no `sys.path` changes):

```python
import cv2

from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
from samples.vision.paddle_ocr.runtime.python.model_runner import create_stage_runners
from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline

pair = resolve_pair(
    "s100",
    det_asset_id="s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm",
    rec_asset_id="s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm",
    det_model_path="/opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm",
    rec_model_path="/opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm",
)
detector, recognizer = create_stage_runners(pair, priority=0, bpu_cores=[0])
image = cv2.imread("samples/vision/paddle_ocr/test_data/s100/gt_2322.jpg")
result = OCRPipeline(pair, detector, recognizer).predict(image)
print(result.texts)
```

`resolve_pair` binds exact assets and validates target/references/dictionary
identity; `create_stage_runners` returns two lazy stage runners sharing the
scheduling parameters; `OCRPipeline.predict(image)` chains
detection → crop → recognition and returns the result object (`texts`,
`boxes`).

<a id="stage-io"></a>
## Stage I/O

This sample is the repository's multi-stage reference: each stage has an
explicit data contract and the pipeline composes them in readable order.

| Stage | Input | Output | Contract |
| --- | --- | --- | --- |
| `tensor_io.prepare_detection` (det pre) | BGR image | packed NV12 `[1,960,640,1]` U8 (X5, linear resize) or split `x_y [1,640,640,1]` + `x_uv [1,320,320,2]` U8 (S100, area resize) | no CLI, no download, no file writes |
| detector `forward` | the NV12 tensors above | raw F32 score map (e.g. `[1,1,640,640]`) | lazy `HB_HBMRuntime` load; metadata validated against the binding; no decode here |
| det `post_process` | raw score map, per-call geometry context | thresholded contours → ordered polygon boxes and crops | threshold `0.5`; dilation and minimum-area filtering are target-local; the context carries this call's resize/scale only |
| `tensor_io.prepare_recognition` (rec pre) | BGR crop | RGB F32 NCHW `[1,3,48,320]` in `[0,1]`, linear resize | same discipline as detection pre |
| recognizer `forward` | the RGB tensor above | F32 `[1,40,C]` (`C=97` X5, `C=18710` S100) | lazily loads the second stage; physical tensors validated |
| rec `post_process` | raw `[1,40,C]`, dictionary from the pair | decoded strings, one per crop | CTC best path: blank reset, repeat collapse, dictionary order; no file I/O |
| `OCRPipeline.predict` | BGR image | result with aligned `boxes`/`texts` | chains exactly the stages above; no second algorithm |

Stage-error ownership is part of the contract: a detection-stage failure is
attributed to the detector (artifact, metadata, target), a recognition-stage
failure to the recognizer — the pipeline does not blur them. Zero detections
is a valid outcome that skips recognition. Per-call geometry lives in the
prepared context, never in instance fields reused across calls, so
interleaved images of different sizes cannot contaminate each other (covered
by host tests).

<a id="troubleshooting"></a>
## Troubleshooting

- **No audited pair / mixed references:** run `--list-models`; take both
  references from one row and the pair's dictionary.
- **Model file not found:** run `--prepare` or copy an existing artifact,
  then pass both paths explicitly; the inference path never fetches.
- **Runtime metadata mismatch:** inspect the artifact with the toolchain's
  model-info command. X5 requires packed NV12 and 97 classes; S100 requires
  split NV12 and 18,710 classes.
- **`pyclipper` missing:** install it in the board Python environment
  before processing a non-empty detector result; help/list/dry-run do not
  need it.
- **No boxes:** verify the image and detector artifact, then try the
  observed `0.5` threshold; a threshold change is a runtime choice, not a
  new accuracy result.
- **Garbled text:** use the dictionary paired with the recognizer. S100's
  blank/line/space order must remain unchanged; it cannot use the X5
  alphabet.
