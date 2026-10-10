English | [简体中文](README_cn.md)

# X5 C++ depth runtime

Native inference for the five published X5 YOLO26 Depth BINs. The named
`Yolo26Depth` model owns preprocessing, the warmup plus one timed forward, and
the calibrated log-depth restore; parsing, rendering and report IO live in the
CLI module. The binary accepts flag and three-positional-argument command
forms, and writes NumPy-headed output arrays for direct offline evaluation.

<a id="overview"></a>
## C++ inference

Estimate relative depth on X5 from a BGR image. `main.cpp` visibly constructs
`Yolo26Depth` and calls `predict`; stage math, tensor contract and SDK
ownership live in `depth.cpp`; parsing, colorization and artifact/report
writing live in `cli.cpp`. The launcher selects the artifact and starts the
native program.

<a id="directory"></a>
## Directory structure

```text
cpp/
├── inc/
│   ├── cli.hpp  # CLI options, image loading, colorization and report declarations
│   └── depth.hpp  # Yolo26Depth model, stage-data types, tensor contract
├── src/
│   ├── main.cpp  # thin entry: parse options, construct model, predict, save
│   ├── cli.cpp  # option parsing, depth colorization, NPY/F32/PNG/report writing
│   └── depth.cpp  # preprocess/infer/postprocess, letterbox/NV12, SDK handles
├── tests/  # (sample tests/) native contract, resources and CLI host tests
├── CMakeLists.txt  # RDK_TARGET-gated build definition
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── launcher.py  # Python script
└── run.sh  # Run the sample
```

<a id="supported-boards"></a>
## Supported boards and validation

| Target | Variants | Native status |
|---|---|---|
| X5 | n/s/m/l/x, calibrated log-depth / NV12 | implemented |
| S100/S100P/S600 | use the Python runtime | no native depth implementation; this binary explicitly refuses S targets |

Identity follows the repository registry: boardinfo `x5`; otherwise socinfo
`x5u/x5h/x5m`; otherwise exact device-tree model `D-Robotics RDK X5 V1.0`.
A present unknown boardinfo/socinfo value prevents fallback. The constructor's
identity gate runs before any SDK call, and a model path never bypasses it.

<a id="dependencies"></a>
## Dependencies

Use the matching X5 Linux SDK with `dnn/hb_dnn.h`, `dnn/hb_sys.h`, `libdnn`,
C++17 compiler, CMake ≥3.16, OpenCV core/imgproc/imgcodecs, pthread, rt and dl.
The build links the system DNN/OpenCV libraries; build against the real SDK
include/library paths when compiling for the board. Fake test headers are never
release include paths.

The launcher/model downloader additionally need Python and PyYAML for the shared
manifest tooling. They do not need `hbm_runtime`; the binary uses native DNN.
No script installs packages, downloads a model during inference, or suppresses
unresolved linker symbols. Prepare dependencies explicitly in the target SDK
environment. Commands below run from repository root.

<a id="build"></a>
## Model preparation and build

```bash
bash samples/vision/yolo26_depth/model/download.sh --target x5 --variant n
bash samples/vision/yolo26_depth/runtime/cpp/run.sh --target x5 --variant n \
  --build --output /work/depth/cpp-n
```

`--build` validates local identity, selected artifact and input before invoking
CMake in `runtime/cpp/build/x5`, then executes the built binary. X5 published
assets require their manifest SHA-256. Existing output directories are refused.
To compile separately in the correctly provisioned SDK environment:

```bash
cmake -S samples/vision/yolo26_depth/runtime/cpp \
  -B samples/vision/yolo26_depth/runtime/cpp/build/x5 \
  -DRDK_TARGET=x5 -DCMAKE_BUILD_TYPE=Release
cmake --build samples/vision/yolo26_depth/runtime/cpp/build/x5 --parallel 2
```

CMake requires an explicit `RDK_TARGET` of `x5` (auto is refused while
cross-compiling because it would read the build host's SoC identity) and
accepts its standard `DNN_INCLUDE_DIR` and `DNN_LIBRARY` cache overrides
for an explicitly prepared SDK. A standalone successful compile is not proof of
board behavior or runtime compatibility.

<a id="run"></a>
## Run or inspect without execution

```bash
bash samples/vision/yolo26_depth/runtime/cpp/run.sh --list-models
bash samples/vision/yolo26_depth/runtime/cpp/run.sh --target x5 --variant l --dry-run
bash samples/vision/yolo26_depth/runtime/cpp/run.sh --target x5 --variant l \
  --test-img samples/vision/yolo26_depth/test_data/bus.jpg \
  --warmup 3 --output /work/depth/cpp-l
```

List/dry-run need no model file, SDK, CMake or board and never execute a build.
Normal runs use the already-built binary unless `--build` is explicit.
`run.sh` resolves relative user paths from repository root.

Direct binary invocation is also available, including the three-positional-argument
form:

```bash
samples/vision/yolo26_depth/runtime/cpp/build/x5/yolo26_depth \
  --model-path /work/depth/model.bin --test-img /work/depth/input.jpg \
  --output /work/depth/native-direct --target x5
# Equivalent positional form:
samples/vision/yolo26_depth/runtime/cpp/build/x5/yolo26_depth \
  /work/depth/model.bin /work/depth/input.jpg /work/depth/native-positional
```

The direct binary verifies board/tensor contracts, not publisher identity.
Use the launcher when you need manifest hash verification and `launch-report.json`.
For a self-compiled model, combine `--converted-model`, `--model-path` and an exact
`--asset-id` from list-models. This selects the X5 calibrated-log/NV12 tensor
contract for the local model.

<a id="parameters"></a>
## Parameters

| Entry | Option | Meaning / default |
|---|---|---|
| launcher | `--target`, `--variant`, `--asset-id` | x5 by default, n unless inferred from exact ID; only X5 may execute |
| launcher | `--model-path`, `--converted-model` | external model with exact ID; custom mode explicitly separates provenance |
| both | `--test-img`, `--output` | source image and new output directory; launcher defaults to bundled bus / `outputs/yolo26_depth_cpp` |
| both | `--warmup` | nonnegative forward calls before one timed call; binary default 0 (Python runtime default 3) |
| launcher | `--binary` | explicit prebuilt binary; incompatible with `--build` |
| launcher | `--build` | explicit configure/build/run after prerequisite gates |
| launcher | `--dry-run`, `--list-models` | mutually exclusive host inspection modes |
| binary | `--model-path` / `--model`, `--test-img` / `--input` | explicit model and image; no implicit model selection |
| binary | `--help` | print usage without loading the SDK model |

The binary takes space-separated `--key value` pairs only; `--key=value` is
not accepted. Scheduling uses the SDK defaults; no core/priority controls
are exposed. Invalid flags, inputs or target fail with exit code 2.

<a id="interface-lifecycle"></a>
## API, stages and resource lifetime

```cpp
#include "depth.hpp"
#include <opencv2/imgcodecs.hpp>

yolo26_depth::Yolo26Depth model("/work/depth/model.bin",
                                yolo26_depth::DepthOptions{/*warmup=*/3});
cv::Mat image = cv::imread("/work/depth/input.jpg");
auto prepared = model.preprocess(image);       // owned NV12 + letterbox context
auto raw = model.infer(prepared);              // 3 warmups + 1 timed forward
auto result = model.postprocess(raw, prepared.context);
// model.predict(image) runs exactly these same three stages.
```

`infer` performs exactly `DepthOptions::warmup` unmeasured forwards followed by
one timed forward and returns the raw calibrated log-depth beside a
`RunMetadata{latency_ms, warmup}`; the timer brackets the full forward only
(buffer copy, cache operations, SDK run, raw output copy), never the restore.
Each call returns owned data that survives the next predict. The model is
noncopyable; use separate instances for concurrent clients and make no SDK
concurrency claim. The optional C++ execution-gate constructor argument is a
host-test injection seam, not a command-line bypass.

| Stage | Contract |
|---|---|
| preprocess | nonempty BGR CV_8UC3 → 768 linear letterbox, padding 114 → owned flat NV12 bytes plus the per-call ImageContext |
| infer | exact warmup + one timed forward; one input/model/output; returns owned raw F32 192-square values with RunMetadata; no exp, rendering or file IO |
| postprocess | finite 192×192 F32 → exp, linear resize to 768, crop using supplied context, restore original size; rejects mismatched contexts |
| predict | invokes exactly preprocess → infer → postprocess on one image |

Geometry uses Python-compatible ties-to-even; extreme aspect ratios that collapse
one resized dimension fail clearly. SDK output must be F32/NONE, NHWC or NCHW
single-channel 192-square. Positive byte strides or aligned-shape-derived strides
are validated against allocation capacity; padded outputs are copied correctly.
Input requires compact NV12 pyramid geometry; padded logical input geometry is
explicitly unsupported rather than silently copied incorrectly.

The model releases packed handles, allocated buffers and the finished task
exactly once on the success path (a failed release is reported), with a guard
covering every exceptional path; destruction frees only acquired resources, so a
failed constructor leaks nothing. Rendering and all artifact/report IO live in
`cli.cpp`; `main.cpp` only parses, constructs the model, predicts and saves.

<a id="results-interpretation"></a>
## Results and verification limits

Successful native execution writes:

- `log_depth.npy`: owned float32 192×192 calibrated log-depth.
- `depth_native.npy`: float32 original H×W relative depth.
- `depth_native.f32`: the same depth as little-endian row-major F32 without header;
  read dimensions from the report.
- `depth.png`, `overlay.png`: inverted TURBO with interpolated 2%/98% percentiles;
  overlay weights original 0.45 / depth color 0.55.
- `report.json`: actual model name, paths, shapes, warmup and timing; SDK version
  remains unknown unless separately recorded.

The launcher adds `launch-report.json` with exact command, UTC bounds, return
code, model/input/binary/native-report hashes and published/custom provenance,
plus native stdout/stderr logs when an output directory was created. On an early
failure output may exist only on stderr; a partial directory is not success.

`report.json`'s `latency_ms` covers one complete forward including buffer copies,
cache operations, SDK calls and raw output copying after exactly `warmup`
unmeasured forwards. It is not HRT BPU-only timing. Depth is relative,
colors are not metres, and depth quality is evaluated with the dataset metrics.
