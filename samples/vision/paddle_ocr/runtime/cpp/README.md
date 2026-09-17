English | [简体中文](./README_cn.md)

# PaddleOCR C++ runtime

This directory contains the maintained native two-stage runtime for the
audited S100 PP-OCRv6 pair. It runs DB text detection, crops the detected
regions, applies CRNN+CTC recognition, and writes a side-by-side JPEG with
boxes on the left and recognized text on the right. The C++ implementation
retains the source S16 detector-output path for older compatible artifacts and
the F32 path used by the checked-in PP-OCRv6 artifact. It also retains the
original FreeType font option and defaults.

The Python runtime is the portable library entrypoint for both audited pairs:
X5 uses PP-OCRv3 with a packed NV12 tensor and a 97-class output; S100 uses
PP-OCRv6 with split NV12 tensors and a 18,710-class dictionary. The native
implementation in this directory is S-series C++ code and is not an X5 C++
port. S100P/S600 source-level SOC defaults remain in the compatibility build,
but this integration has no new board evidence for those targets.

## Prepare the board

Build on an RDK S100 image with the existing DNN/HB UCP, OpenCV, gflags,
polyclipping, and FreeType development files. The launcher does not install
packages or download models. On a Debian-based development image, the missing
headers can be installed explicitly:

```bash
sudo apt install libgflags-dev libpolyclipping-dev libfreetype6-dev
```

Some newer images name the FreeType package `libfreetype-dev`. Use the package
name provided by that image. The matching RDK SDK libraries under
`/usr/hobot` and the C++ headers under `/usr/hobot/include` are required.

Prepare the published S100 artifacts explicitly from a full checkout if they
are not already under `/opt/hobot/model/s100/basic`:

```bash
python samples/vision/paddle_ocr/runtime/python/main.py --prepare \
  --target s100 \
  --det-asset-id s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec-asset-id s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --model-dir /opt/hobot/model/s100/basic
```

The command uses the existing release manifest and prints the observed file
digests. It is the only network-capable preparation operation in the sample.
The bundled fixture image and dictionary are copied into the canonical
checkout. The default font is kept at its historical path; the run script
resolves the existing S font fixture automatically.

## Build and run

From any directory in a full checkout, run:

```bash
bash samples/vision/paddle_ocr/runtime/cpp/run.sh
```

The launcher configures CMake in `runtime/cpp/build`, builds `paddle_ocr`, and
executes it with absolute paths to the canonical S100 fixture, dictionary, and
font. It passes user arguments after those defaults, so an explicit flag takes
precedence:

```bash
bash samples/vision/paddle_ocr/runtime/cpp/run.sh -- \
  --det_model_path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec_model_path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --test_image /data/sign.jpg \
  --label_file /data/ppocrv6_dict.txt \
  --font_path /data/NotoSansCJK-Regular.ttc \
  --img_save_path /data/sign_result.jpg
```

For a manual build, use the same source directory and pass paths directly to
the executable:

```bash
cmake -S samples/vision/paddle_ocr/runtime/cpp \
      -B samples/vision/paddle_ocr/runtime/cpp/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build samples/vision/paddle_ocr/runtime/cpp/build --parallel
samples/vision/paddle_ocr/runtime/cpp/build/paddle_ocr \
  --det_model_path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec_model_path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --test_image samples/vision/paddle_ocr/test_data/s100/gt_2322.jpg \
  --label_file samples/vision/paddle_ocr/test_data/s100/ppocrv6_dict.txt \
  --font_path platforms/s/samples/vision/paddle_ocr/test_data/FangSong.ttf
```

The flags are `--det_model_path`, `--rec_model_path`, `--test_image`,
`--label_file`, `--threshold` (default `0.5`), `--ratio_prime` (default
`2.7`), `--font_path`, and `--img_save_path` (default `result.jpg`). The
dictionary is read one line at a time, with a blank class prepended and one
trailing space class appended. This preserves the 18,710-class PP-OCRv6
contract, including dictionary lines containing punctuation.

## Result and library use

The executable prints one prediction per retained crop and writes
`img_save_path`. The left panel contains the original image and ordered
minimum-area boxes. The right panel is a white canvas rendered with the
recognized strings. A missing or unreadable font is reported by the existing
visualization utility; pass a known TTF/ TTC file with `--font_path`.

The native API is declared in
[`inc/paddle_ocr.hpp`](./inc/paddle_ocr.hpp). The two wrappers expose
`PaddleOCRDet::init`, `pre_process_det`, `infer`, `post_process_det`, and the
corresponding `PaddleOCRRec`/CTC functions for applications that need to
compose the stages. `src/paddle_ocr.cpp` keeps model metadata, stride-aware
tensor access, NV12 preparation, S16/F32 detector output handling, geometry,
and CTC decoding together with the existing C++ utility contract.

The Python library is usually easier to embed:

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
result = OCRPipeline(pair, detector, recognizer).predict(cv2.imread("sign.jpg"))
print(result.texts)
```

## Conversion and evaluation

The complete, target-separated conversion commands are in
[`conversion/README.md`](../../conversion/README.md). X5 PP-OCRv3 uses the
`hb_mapper`/`bayes-e` recipes; S100 PP-OCRv6 uses the `hb_compile`/`nash-e`
recipes. Their dictionaries, model graphs, output names, and NV12 protocols
are separate. The conversion notes call out the S100 trailing `Dequantize`
requirement that keeps detector output F32.

The executable JSON result can be evaluated against a labeled JSONL record by
[`evaluator/evaluate.py`](../../evaluator/evaluate.py). Evaluation reports
IoU-matched detection counts and recognition agreement for the records supplied
by the user; it does not turn the bundled demonstration image into an accuracy
claim. See [`evaluator/README.md`](../../evaluator/README.md) for the record
format and empty-ground-truth behavior.

## Troubleshooting

* `hbm_runtime` or DNN initialization fails: run on the matching RDK image and
  verify both model files exist. Help/list/dry-run Python modes remain
  available without the board SDK.
* Model metadata or tensor names do not match: do not mix X5 PP-OCRv3 and S100
  PP-OCRv6 artifacts. Use the pair's exact detector and recognizer references.
* No boxes are returned: check the image path, detector threshold, and that the
  model output is the expected F32 map. The C++ `--threshold` default is `0.5`.
* Text is garbled: use the dictionary shipped with the same S100 PP-OCRv6
  recognizer. The dictionary is not interchangeable with the X5 alphabet.
* CMake cannot find `polyclipping`, gflags, or FreeType: install the matching
  development headers on the build image and rerun CMake; the launcher does
  not modify the system.

Host Python checks and the migration evidence are described in the parent
[`README.md`](../../README.md). Board accuracy and performance require a
labeled dataset and a target-specific measurement run; neither is inferred
from this sample's deterministic pipeline comparison.
