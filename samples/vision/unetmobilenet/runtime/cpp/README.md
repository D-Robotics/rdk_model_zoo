English | [简体中文](README_cn.md)

# UNetMobileNet C++ runtime

<a id="overview"></a>
## C++ inference

Use this directory for c++ inference. `main.cpp` visibly constructs the named
model and calls `predict`; stage math, tensor contract and SDK ownership live
in `segment.cpp`; parsing, overlay rendering and artifact/report writing live
in `cli.cpp`.

<a id="directory"></a>
## Directory structure

```text
cpp/
├── inc/
│   ├── cli.hpp  # CLI options, image loading, overlay and report declarations
│   └── segment.hpp  # UnetMobileNet model, score contract, board identity
├── src/
│   ├── main.cpp  # thin entry: parse options, construct model, predict, save
│   ├── cli.cpp  # option parsing, overlay rendering, image/mask/report writing
│   └── segment.cpp  # preprocess/infer/postprocess, NV12 binding, decode, SDK handles
├── tests/  # Automated tests (score contract + resource cleanup with fake SDK headers)
├── CMakeLists.txt  # RDK_TARGET-gated build definition
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── launcher.py  # Python script
└── run.sh  # Run the sample
```

<a id="supported-boards"></a>
## Supported boards

S100/S600 have distinct published HBM files; Python and C++ use the same exact target-scoped selection. X5/S100P have no asset and fail. Native binaries embed the explicit build target and independently check local S identity, including s100 + board_type s100p/RDK S100P refinement.

<a id="dependencies"></a>
## Dependencies

C++17 compiler, CMake 3.16+, OpenCV core/imgproc/imgcodecs development libraries, and the matching board DNN/UCP SDK. Include roots: /usr/hobot/include, /usr/include/hobot and /usr/include/hobot/dnn; libraries under /usr/hobot/lib. Launcher uses Python 3.10+ and PyYAML for selection. Use the S SDK provided by your board image; perform the full SDK build in that environment.

<a id="build"></a>
## Build

```bash
# cwd: repository root, on S100; install development dependencies explicitly
sudo apt install build-essential cmake libopencv-dev
bash samples/vision/unetmobilenet/model/download.sh --target s100
bash samples/vision/unetmobilenet/runtime/cpp/run.sh --target s100 --build
# Subsequent run, same binary/model:
bash samples/vision/unetmobilenet/runtime/cpp/run.sh --target s100
```

The launcher verifies board identity, prepared model and input file BEFORE starting CMake. It never installs or downloads implicitly. Separate build/s100 and build/s600 directories prevent accidental reuse; CMake requires an explicit RDK_TARGET of s100 or s600 (auto is refused while cross-compiling because it would read the build host's SoC identity), normalizes the SoC string and fails on anything else rather than guessing a macro.

<a id="run"></a>
## Run

```bash
# cwd: repository root; board DNN/UCP headers and libraries already present
cmake -S samples/vision/unetmobilenet/runtime/cpp -B samples/vision/unetmobilenet/runtime/cpp/build/s600 -DRDK_TARGET=s600 -DCMAKE_BUILD_TYPE=Release
cmake --build samples/vision/unetmobilenet/runtime/cpp/build/s600 --parallel 2
bash samples/vision/unetmobilenet/model/download.sh --target s600
bash samples/vision/unetmobilenet/runtime/cpp/run.sh --target s600 --alpha-f 0.5 --img-save-path outputs/unetmobilenet/native.png --mask-save-path outputs/unetmobilenet/native_labels.png --report-path outputs/unetmobilenet/native.json
```

After an explicit build and preparation, run.sh with no arguments works on the matching recognized board. --help/list-models/dry-run are SDK-free launcher modes. Direct binary invocation requires --target, --model-path and --test-img; all paths supplied to the binary are literal paths, not manifest resolution.

<a id="parameters"></a>
## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--target` | `auto` | launcher detects exact board; native binary requires explicit s100/s600 |
| `--asset-id` | `None` | launcher exact asset identity; required with external model-path |
| `--model-path` | `None` | launcher resolves model/<target>/ HBM; native binary requires path |
| `--test-img` | `samples/vision/unetmobilenet/test_data/segmentation.png` | launcher default; native binary requires path |
| `--img-save-path` | `result.jpg` | original-resolution overlay |
| `--mask-save-path` | `unetmobilenet_mask.png` | lossless uint8 class IDs 0..18;.png required |
| `--report-path` | `unetmobilenet_cpp_report.json` | JSON report |
| `--alpha-f` | `0.75` | original image weight in [0,1] |
| `--priority` | `0` | scheduler priority 0..255 |
| `--bpu-core` | `-1` | any core (default); otherwise core index 0..3 |
| `--build` | `false` | launcher explicitly configures/builds before running |
| `--binary` | `None` | launcher custom executable; incompatible with --build |
| `--list-models` | `false` | launcher manifest listing only |
| `--dry-run` | `false` | launcher prints resolved command; no build/SDK; exclusive with list-models |

The binary accepts kebab-case and the underscore aliases (--model_path, --test_img, --alpha_f); the launcher uses kebab-case only. --help/-h prints help. Output paths use cwd and existing files are replaced.

<a id="interface-lifecycle"></a>
## Interface and lifecycle

`UnetMobileNet` (segment.hpp/segment.cpp) owns the packed/model handles, the two split NV12 input buffers and the score output buffer. Constructing it checks the requested target against the embedded build target (an injectable gate for tests; the default reads /sys/class/boardinfo before any SDK call), validates priority 0..255 and bpu-core -1 or 0..3, then validates input counts, split-NV12 geometry/order, row pitch, byte strides, output rank/type and score capacity/quantization metadata before any allocation. Destruction frees only acquired resources, so a failed constructor leaks nothing.

The public stages are preprocess(image) -> owned Y/UV Mats plus a per-call ImageContext, infer(prepared) -> an owned RawScores copy (padded bytes and descriptor untouched by later calls), and postprocess(raw, context) -> original-resolution CV_32S class IDs; predict(image) composes exactly those three. score_spec() exposes the bound output descriptor. Every SDK result is checked, including submit/wait/release; the finished task is released exactly once on the success path and a task guard covers every exceptional path. The model is noncopyable and not thread-safe; use separate instances per thread.

`cli.cpp` owns rendering and report IO only: parse_options/print_help, load_image, render_overlay (19-color palette, alpha blend) and save_results (overlay, uint8 PNG mask, JSON report). main.cpp constructs the model, calls predict once and hands the result to save_results; no model or SDK work lives in the CLI module.

<a id="results-interpretation"></a>
## Results

Success returns 0, errors 2. result.jpg is an overlay on the original image; the PNG mask contains uint8 IDs, while the API mask remains int32. Compare PNG values with Python NPY labels after reading both as integer arrays. JSON/stdout records target, asset_id, model_path, input_path, publisher_sha256, runtime_version (unknown when not queried), mask_shape, score_shape, score_dtype, scaled, alpha_f, priority, bpu_core and output paths. Masks are restored directly from the score grid to the original size with nearest-neighbor integer mapping.
