English | [简体中文](README_cn.md)

# PaddleOCR C++ runtime (S series)

<a id="overview"></a>
## C++ inference

The native two-stage runtime for the S100 PP-OCRv6 pair:
DB text detection, region cropping, CRNN+CTC recognition, and a side-by-side
JPEG rendering (original image with ordered boxes on the left, recognized
text on the right). The detector supports S16 and F32 outputs; the bundled PP-OCRv6 artifact uses F32. Configure rendering with the FreeType font option.

<a id="directory"></a>
## Directory structure

```text
cpp/
├── inc/
│   ├── ocr.hpp   # PaddleOCRDet/PaddleOCRRec/PaddleOCR classes and owned stage-data types
│   └── cli.hpp   # CLI options and helpers
├── src/
│   ├── ocr.cpp   # runtime lifecycle, det/rec preprocess/infer/postprocess stages
│   ├── cli.cpp   # argument parsing, defaults, dictionary loading, rendering
│   └── main.cpp  # entry point: parse options, predict, print and save
├── CMakeLists.txt  # build (C++17, explicit RDK_TARGET board selection)
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── run.sh  # Build and launch the sample
```

<a id="supported-boards"></a>
## Supported boards

| Board | Status |
| --- | --- |
| S100 | supported |
| S600 | supported |
| S100P | not-supported (no PaddleOCR pair published) |
| X5 | not-supported (no X5 C++ implementation provided; use the Python runtime) |

<a id="dependencies"></a>
## Dependencies

Build on an RDK S image with: CMake and a C++17 compiler; OpenCV development
files; `polyclipping` and FreeType development libraries; the
Horizon DNN/UCP headers under `/usr/hobot/include` and libraries under
`/usr/hobot/lib`. On a Debian-based development image the missing headers
can be installed explicitly (some newer images name the FreeType package
`libfreetype-dev` — use the name that image provides):

```bash
sudo apt install libpolyclipping-dev libfreetype6-dev
```

The launcher installs nothing, modifies no SDK, and downloads no models.
The matching artifacts must exist under `/opt/hobot/model/s100/basic` (or be
passed explicitly); prepare them with the Python entrypoint's `--prepare`
when absent (see [model preparation](../../model/README.md#preparation)).

<a id="build"></a>
## Build

The launcher builds automatically; a manual build (cwd: repository root;
success: `paddle_ocr` binary under the build directory):

```bash
cmake -S samples/vision/paddle_ocr/runtime/cpp \
      -B samples/vision/paddle_ocr/runtime/cpp/build \
      -DRDK_TARGET=s100 -DCMAKE_BUILD_TYPE=Release
cmake --build samples/vision/paddle_ocr/runtime/cpp/build --parallel
```

`RDK_TARGET` selects the board (`s100` or `s600`); when configuring natively
on the board it may be omitted (`auto` reads the on-board SoC identity).
Cross compilation must pass an explicit target.

<a id="run"></a>
## Run

From any directory in a full checkout (inputs: S100 artifact pair, the
checked-in S100 fixture and dictionary, the S font fixture; output: one
prediction line per crop plus the rendered JPEG; success: exit 0):

```bash
bash samples/vision/paddle_ocr/runtime/cpp/run.sh
```

The launcher resolves absolute paths to the fixture, dictionary,
and font, then forwards any user flags after them, so an explicit flag takes
precedence:

```bash
bash samples/vision/paddle_ocr/runtime/cpp/run.sh -- \
  --det-model-path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec-model-path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --test-img /data/sign.jpg \
  --vocabulary-path /data/ppocrv6_dict.txt \
  --font-path /data/NotoSansCJK-Regular.ttc \
  --img-save-path /data/sign_result.jpg
```

A manual invocation passes the paths directly:

```bash
samples/vision/paddle_ocr/runtime/cpp/build/paddle_ocr \
  --det-model-path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec-model-path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --test-img samples/vision/paddle_ocr/test_data/s100/gt_2322.jpg \
  --vocabulary-path samples/vision/paddle_ocr/test_data/s100/ppocrv6_dict.txt \
  --font-path samples/vision/paddle_ocr/test_data/FangSong.ttf
```

<a id="parameters"></a>
## Parameters

Options of the `paddle_ocr` binary, matching the Python runtime's
kebab-case names (the launcher fills the fixture paths unless overridden):

| Option | Default | Meaning |
| --- | --- | --- |
| `--det-model-path` | SoC-dependent: `/opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` (S100) or the `s600` variant | detector HBM |
| `--rec-model-path` | SoC-dependent: `/opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` (S100) or the `s600` variant | recognizer HBM |
| `--test-img` | (launcher: `test_data/s100/gt_2322.jpg`) | BGR input image |
| `--vocabulary-path` | (launcher: `test_data/s100/ppocrv6_dict.txt`) | recognizer dictionary, one entry per line |
| `--threshold` | `0.5` | detector score-map binarization threshold |
| `--ratio-prime` | `2.7` | contour expansion ratio |
| `--font-path` | (launcher: S font fixture `FangSong.ttf`) | TTF/TTC font for text rendering |
| `--img-save-path` | `result.jpg` | rendered side-by-side output path |
| `--help` / `-h` | — | print usage |

The dictionary is read verbatim one line at a time with a blank class
prepended and one trailing space class appended, preserving the
18,710-class PP-OCRv6 contract including dictionary lines containing
punctuation such as `{`, `}` and `,`.

<a id="interface-lifecycle"></a>
## Interface and lifecycle

The public API is declared in [`inc/ocr.hpp`](./inc/ocr.hpp). `main.cpp`
parses the options, constructs `PaddleOCR model(det_path, rec_path)` — the
constructor loads both HBM packs, reads the tensor metadata and allocates
the reusable buffers — then calls
`model.predict(image, dictionary, options)` and renders the returned result.
All DNN/UCP types stay inside `src/ocr.cpp` (private `Impl` structs), so the
header depends only on OpenCV and the standard library.

Each stage is exposed separately and returns data owned by the caller:

- detector: `OcrDetPrepared preprocess(const cv::Mat&)` (INTER_AREA resize
  and NV12 planes), `OcrDetRaw infer(const OcrDetPrepared&)` (the prediction
  map in its own domain — float32 for PP-OCRv6, int16 + scale for the legacy
  S16 export), `TextDetResult postprocess(const OcrDetRaw&, const cv::Mat&,
  const OcrOptions&)` (threshold, contour dilation with
  D' = area × ratio_prime / perimeter, minimum-area boxes and rectified
  crops);
- recognizer: `OcrRecPrepared preprocess(const cv::Mat&)` (RGB, [0, 1]
  float32 CHW planes with no ImageNet normalization — the model was
  calibrated on raw [0, 1] crops), `OcrRecRaw infer(const OcrRecPrepared&)`
  (stride-aware copy of the CTC logits), `std::string postprocess(const
  OcrRecRaw&, const std::vector<std::string>& id2token)` (greedy CTC);
- `PaddleOCR::predict` composes the stages visibly: detection, then one
  recognition pass per crop. No detected text region means no recognition
  calls; a failing crop skips only its own text while the remaining crops
  are still recognized. The result keeps each surviving text's original crop
  index and records every skipped crop with its index and cause, so the CLI
  can report them.

The detector accepts two output domains: a float32 prediction map
(PP-OCRv6 export) or a scale-quantized int16 map with a non-null scale
entry (legacy PP-OCRv3 export). Any other tensor type, an int16 map without
scale quantization, or a float32 map that declares quantization is rejected
as an unsupported contract. The recognizer always reads float32 CTC logits.

Errors surface as C++ exceptions (SDK error descriptions included); the
entry point prints them and exits with status 2. Resources are released
by RAII on every path, including partial initialization failures. There are
no background threads and the process performs one synchronous
detect-then-recognize pass. Inference uses the S-series `hbDNNInferV2` +
`hbUCPMallocCached` API family (S100/S600 only; not call-level compatible
with the X5 C++ API).

<a id="results-interpretation"></a>
## Results interpretation

The executable prints one prediction per retained crop (with its original
crop index) and writes `img_save_path`: the left panel is the original image
with ordered minimum-area boxes, the right panel a white canvas with the
recognized strings. A crop whose recognition failed is reported on stderr
with its original index and cause, and the visualization pairs the surviving
texts with the boxes compactly. A missing or
unreadable font is reported by the visualization utility — pass a known
TTF/TTC with `--font-path`. Success is exit 0.
Record the board identity, artifact references, full build/run commands,
and the printed predictions plus the rendered image for evidence. Run the
S600 commands on an S600 board for S600 results; the S100 run is not a
substitute.
