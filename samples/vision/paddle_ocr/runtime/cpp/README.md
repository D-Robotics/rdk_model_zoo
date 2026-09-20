English | [简体中文](./README_cn.md)

# PaddleOCR C++ runtime (S series)

The maintained native two-stage runtime for the audited S100 PP-OCRv6 pair:
DB text detection, region cropping, CRNN+CTC recognition, and a side-by-side
JPEG rendering (original image with ordered boxes on the left, recognized
text on the right). It retains the source S16 detector-output path for older
compatible artifacts alongside the F32 path used by the checked-in
PP-OCRv6 artifact, and the original FreeType font option and defaults.

<a id="supported-boards"></a>
## Supported boards

| Board | Status |
| --- | --- |
| S100 | supported-verified (built and run 2026-09-17; rendered output pixels equal to the source baseline) |
| S600 | supported-not-run (same sources and SoC detection; board access unavailable) |
| S100P | not-supported (no audited PaddleOCR pair; build defaults do not constitute support) |
| X5 | not-supported (no X5 C++ source in the audited baseline; use the Python runtime) |

<a id="dependencies"></a>
## Dependencies

Build on an RDK S image with: CMake and a C++17 compiler; OpenCV development
files; `gflags`, `polyclipping`, and FreeType development libraries; the
Horizon DNN/UCP headers under `/usr/hobot/include` and libraries under
`/usr/hobot/lib`. On a Debian-based development image the missing headers
can be installed explicitly (some newer images name the FreeType package
`libfreetype-dev` — use the name that image provides):

```bash
sudo apt install libgflags-dev libpolyclipping-dev libfreetype6-dev
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
      -DCMAKE_BUILD_TYPE=Release
cmake --build samples/vision/paddle_ocr/runtime/cpp/build --parallel
```

<a id="run"></a>
## Run

From any directory in a full checkout (inputs: S100 artifact pair, the
checked-in S100 fixture and dictionary, the S font fixture; output: one
prediction line per crop plus the rendered JPEG; success: exit 0):

```bash
bash samples/vision/paddle_ocr/runtime/cpp/run.sh
```

The launcher resolves absolute paths to the canonical fixture, dictionary,
and font, then forwards any user flags after them, so an explicit flag takes
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

A manual invocation passes the paths directly:

```bash
samples/vision/paddle_ocr/runtime/cpp/build/paddle_ocr \
  --det_model_path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec_model_path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --test_image samples/vision/paddle_ocr/test_data/s100/gt_2322.jpg \
  --label_file samples/vision/paddle_ocr/test_data/s100/ppocrv6_dict.txt \
  --font_path platforms/s/samples/vision/paddle_ocr/test_data/FangSong.ttf
```

<a id="parameters"></a>
## Parameters

Native gflags of the `paddle_ocr` binary (the launcher overrides the first
four with absolute canonical paths; defaults are the audited source values):

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--det_model_path` | string | `/opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` | detector HBM |
| `--rec_model_path` | string | `/opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | recognizer HBM |
| `--test_image` | string | (launcher: `test_data/s100/gt_2322.jpg`) | BGR input image |
| `--label_file` | string | (launcher: `test_data/s100/ppocrv6_dict.txt`) | recognizer dictionary, one entry per line |
| `--threshold` | double | `0.5` | detector score-map binarization threshold |
| `--ratio_prime` | double | `2.7` | contour expansion ratio |
| `--font_path` | string | (launcher: S font fixture `FangSong.ttf`) | TTF/TTC font for text rendering |
| `--img_save_path` | string | `result.jpg` | rendered side-by-side output path |

The dictionary is read one line at a time with a blank class prepended and
one trailing space class appended, preserving the 18,710-class PP-OCRv6
contract including dictionary lines containing punctuation.

<a id="interface-lifecycle"></a>
## Interface and lifecycle

The public API is declared in [`inc/paddle_ocr.hpp`](./inc/paddle_ocr.hpp):
`PaddleOCRDet::init` / `pre_process_det` / `infer` / `post_process_det` and
the corresponding `PaddleOCRRec`/CTC functions let applications compose the
stages explicitly. `src/paddle_ocr.cpp` keeps model metadata extraction,
stride-aware tensor access, NV12 preparation, S16/F32 detector-output
handling, geometry, and CTC decoding. Heavy work happens after construction,
not in constructors; DNN/UCP resources are released at scope exit; there are
no background threads and the process performs one synchronous
detect-then-recognize pass. Inference uses the S-series `hbDNNInferV2` +
`hbUCPMallocCached` API family, which is not call-level compatible with the
X5 C++ API — that difference is why no X5 C++ port exists here.

<a id="results-interpretation"></a>
## Results interpretation

The executable prints one prediction per retained crop and writes
`img_save_path`: the left panel is the original image with ordered
minimum-area boxes, the right panel a white canvas with the recognized
strings. A missing or unreadable font is reported by the visualization
utility — pass a known TTF/TTC with `--font_path`. Success is exit 0.
Record the board identity, artifact references, full build/run commands,
and the printed predictions plus the rendered image for evidence. S600
stays `not-run` while the board is unreachable; its result cannot be
substituted with the S100 run.
