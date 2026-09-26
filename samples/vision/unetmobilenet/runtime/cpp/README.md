English | [简体中文](README_cn.md)

# UNetMobileNet C++ runtime

<a id="supported-boards"></a>
## Supported boards

S100/S600 have distinct published HBM files; Python and C++ use the same exact target-scoped selection. X5/S100P have no asset and fail. Native binaries embed the explicit build target and independently check local S identity, including s100 + board_type s100p/RDK S100P refinement. Host tests are not real SDK build or board evidence.

<a id="dependencies"></a>
## Dependencies

C++17 compiler, CMake 3.16+, OpenCV core/imgproc/imgcodecs development libraries, and the matching board DNN/UCP SDK. Include roots follow the source S layout: /usr/hobot/include, /usr/include/hobot and /usr/include/hobot/dnn; libraries under /usr/hobot/lib. gflags/fmt are no longer required by this sample. Launcher uses Python 3.10+ and PyYAML for selection. No S SDK version is pinned by the source; full SDK build remains not-run here.

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

The launcher verifies board identity, prepared model and input file BEFORE starting CMake. It never installs or downloads implicitly. Separate build/s100 and build/s600 directories prevent accidental reuse; CMake requires UNETMOBILENET_TARGET and does not infer a macro from an unknown board.

<a id="run"></a>
## Run

```bash
# cwd: repository root; board DNN/UCP headers and libraries already present
cmake -S samples/vision/unetmobilenet/runtime/cpp -B samples/vision/unetmobilenet/runtime/cpp/build/s600 -DUNETMOBILENET_TARGET=s600 -DCMAKE_BUILD_TYPE=Release
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
| `--mask-save-path` | `unetmobilenet_mask.png` | lossless uint8 class IDs 0..18; .png required |
| `--report-path` | `unetmobilenet_cpp_report.json` | JSON report |
| `--alpha-f` | `0.75` | original image weight in [0,1] |
| `--priority` | `0` | scheduler priority 0..255 |
| `--bpu-core` | `-1` | source native default any core; otherwise core index 0..3 |
| `--build` | `false` | launcher explicitly configures/builds before running |
| `--binary` | `None` | launcher custom executable; incompatible with --build |
| `--list-models` | `false` | launcher manifest listing only |
| `--dry-run` | `false` | launcher prints resolved command; no build/SDK; exclusive with list-models |

Native CLI additionally accepts historical underscore spellings (--model_path, --test_img, --alpha_f); the launcher uses kebab-case only. Python source defaults to bpu-cores [0], while C++ source used any core; that deliberate default distinction is preserved. --help/-h prints help. Output paths use cwd and existing files are replaced.

<a id="interface-lifecycle"></a>
## Interface and lifecycle

ModelRunner owns packed/model handles, two input buffers and one output buffer. Constructor validates target, counts, shapes, strides, capacity and score quantization before allocation. Destruction frees only acquired resources; the per-forward task guard releases tasks on submit/wait/cache errors. The runner is noncopyable and not thread-safe; use separate instances per thread. Its optional explicit execution gate is a host-test seam, never a CLI identity bypass.

UnetMobileNetTask accepts a RawRunner callback. pre_process(image) returns owned Y/UV Mats and per-call ImageContext; forward(prepared) returns owned RawScores with padded bytes/metadata unchanged; post_process(raw, context) returns original-resolution CV_32S IDs; predict composes those stages. main.cpp shows complete wiring. Rendering is in visualization.cpp, resource code in model_runner.cpp, and stride/affine decoding in tensor_contract.cpp. No display or file IO belongs in the task.

<a id="results-interpretation"></a>
## Results

Success returns 0, errors 2. result.jpg is the source-color overlay; the PNG mask contains uint8 IDs, while the API mask remains int32. Compare PNG values with Python NPY labels after reading both as integer arrays. JSON/stdout records target, asset_id, model_path, input_path, publisher_sha256, runtime_version (unknown when not queried), mask_shape, score_shape, score_dtype, scaled, alpha_f, priority, bpu_core and output paths. This is a single-image result, not a performance or dataset benchmark. Direct nearest restoration matches canonical Python; source C++ previously resized through model input size, which can differ for non-divisor output dimensions.
