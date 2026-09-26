[English](README.md) | [简体中文](README_cn.md)

# X5 C++ depth runtime

Native inference for the five published X5 YOLO26 Depth BINs. SDK ownership is
separate from the task's preprocessing, forward, postprocessing and predict API.
The source's native capability and positional command form are retained; output
arrays now also have NumPy headers for direct offline evaluation.

<a id="supported-boards"></a>
## Supported boards and validation

| Target | Variants | Native status |
|---|---|---|
| X5 | n/s/m/l/x, calibrated log-depth / NV12 | implemented; host pure-code and fake-SDK tests pass; real SDK build and board inference not-run |
| S100/S100P/S600 | use the Python runtime | no source native depth implementation; this binary explicitly refuses S targets |

Identity follows the repository registry: boardinfo `x5`; otherwise socinfo
`x5u/x5h/x5m`; otherwise exact device-tree model `D-Robotics RDK X5 V1.0`.
A present unknown boardinfo/socinfo value prevents fallback. Both the launcher
and the actual SDK owner check local identity. A model path never bypasses it.

<a id="dependencies"></a>
## Dependencies

Use the matching X5 Linux SDK with `dnn/hb_dnn.h`, `dnn/hb_sys.h`, `libdnn`,
C++17 compiler, CMake ≥3.16, OpenCV core/imgproc/imgcodecs, pthread, rt and dl.
The source linked system DNN/OpenCV libraries; this migration has not compiled
or linked against a real SDK. Fake test headers are never release include paths.

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
  -B samples/vision/yolo26_depth/runtime/cpp/build/x5 -DCMAKE_BUILD_TYPE=Release
cmake --build samples/vision/yolo26_depth/runtime/cpp/build/x5 --parallel 2
```

CMake accepts its standard `DNN_INCLUDE_DIR` and `DNN_LIBRARY` cache overrides
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

Direct binary invocation is also available, including the original three
positional arguments:

```bash
samples/vision/yolo26_depth/runtime/cpp/build/x5/yolo26_depth \
  --model-path /work/depth/model.bin --test-img /work/depth/input.jpg \
  --output /work/depth/native-direct --target x5
# Equivalent source-style command:
samples/vision/yolo26_depth/runtime/cpp/build/x5/yolo26_depth \
  /work/depth/model.bin /work/depth/input.jpg /work/depth/native-legacy
```

The direct binary verifies board/tensor contracts, not publisher identity.
Use the launcher when you need manifest hash verification and `launch-report.json`.
For a self-compiled model, combine `--converted-model`, `--model-path` and an exact
`--asset-id` from list-models. That ID selects a **contract reference**, not a claim
that the new bytes are the published model. It still requires X5 and the same
calibrated-log/NV12 tensor contract.

<a id="parameters"></a>
## Parameters

| Entry | Option | Meaning / default |
|---|---|---|
| launcher | `--target`, `--variant`, `--asset-id` | x5 by default, n unless inferred from exact ID; only X5 may execute |
| launcher | `--model-path`, `--converted-model` | external model with exact ID; custom mode explicitly separates provenance |
| both | `--test-img`, `--output` | source image and new output directory; launcher defaults to bundled bus / `outputs/yolo26_depth_cpp` |
| both | `--warmup` | nonnegative forward calls before one timed call; native default 0, preserving source (Python default is 3) |
| launcher | `--binary` | explicit prebuilt binary; incompatible with `--build` |
| launcher | `--build` | explicit configure/build/run after prerequisite gates |
| launcher | `--dry-run`, `--list-models` | mutually exclusive host inspection modes |
| binary | `--model-path` / `--model`, `--test-img` / `--input` | explicit model and image; no implicit model selection |
| binary | `--help` | print usage without loading the SDK model |

X5 source C++ used SDK scheduling defaults; no unverified core/priority controls
are added here. Invalid flags, inputs or target fail with exit code 2.

<a id="interface-lifecycle"></a>
## API, stages and resource lifetime

```cpp
#include "model_runner.hpp"
#include "yolo26_depth.hpp"
#include <opencv2/imgcodecs.hpp>

yolo26_depth::ModelRunner runner("/work/depth/model.bin");
yolo26_depth::Yolo26DepthTask task(
    [&runner](const auto& nv12) { return runner.run(nv12); });
cv::Mat image = cv::imread("/work/depth/input.jpg");
auto prepared = task.pre_process(image);
auto raw = task.forward(prepared.nv12);
auto result = task.post_process(raw, prepared.context);
// task.predict(image) runs exactly these same three stages.
```

`ModelRunner` must outlive the task callback. Each call returns owned raw F32
values; results own their OpenCV arrays. Do not use one runner concurrently.
Separate runner instances are needed for concurrent clients; no SDK concurrency
claim is made. The optional C++ execution-gate callback is a host-test injection
seam, not a command-line bypass.

| Stage | Contract |
|---|---|
| pre_process | nonempty BGR CV_8UC3 → 768 linear letterbox, padding 114 → owned flat NV12 bytes; immutable-by-convention context returned per call |
| forward | one input/model/output; SDK invocation and structural checks; returns raw calibrated log-depth with no exp, rendering or file IO |
| post_process | 192×192 finite F32 values → exp, linear resize to 768, crop using supplied context, restore original size |
| predict | invokes the same stages; no warmup or timing embedded in task logic |

Geometry uses Python-compatible ties-to-even; extreme aspect ratios that collapse
one resized dimension fail clearly. SDK output must be F32/NONE, NHWC or NCHW
single-channel 192-square. Positive byte strides or aligned-shape-derived strides
are validated against allocation capacity; padded outputs are copied correctly.
Input requires compact NV12 pyramid geometry; padded logical input geometry is
explicitly unsupported rather than silently copied incorrectly.

The SDK owner releases packed models, allocated buffers and per-call task handles
on normal and error paths. Rendering lives in `image_io.cpp`; CLI/timing and
serialization live in `main.cpp`/`cli_io.cpp`. The old combined `Yolo26Depth::Infer`
C++ API is replaced by runner plus task; the archived source retains the old API.

<a id="results-interpretation"></a>
## Results and verification limits

Successful native execution writes:

- `log_depth.npy`: owned float32 192×192 calibrated log-depth.
- `depth_native.npy`: float32 original H×W relative depth.
- `depth_native.f32`: the same depth as little-endian row-major F32 without header,
  preserving the source output capability; read dimensions from the report.
- `depth.png`, `overlay.png`: inverted TURBO with interpolated 2%/98% percentiles;
  overlay weights original 0.45 / depth color 0.55.
- `report.json`: actual model name, paths, shapes, warmup and timing; SDK version
  remains unknown unless separately recorded.

The launcher adds `launch-report.json` with exact command, UTC bounds, return
code, model/input/binary/native-report hashes and published/custom provenance,
plus native stdout/stderr logs when an output directory was created. On an early
failure output may exist only on stderr; a partial directory is not success.

Timing covers one complete forward including buffer copies, cache operations,
SDK calls and raw output copying. It is not source HRT BPU-only timing. Depth is
relative, colors are not metres, and a plausible image is not an accuracy test.

Host tests compile pure geometry/tensor/CLI/serialization code and the real owner
against intentionally minimal fake SDK headers. They exercise thirteen injected
SDK failure points, owned raw output, invalid metadata, padded output strides,
identity precedence and NumPy loading of native files. **Real SDK/OpenCV build,
full native image pipeline, board inference and performance remain not-run.**
