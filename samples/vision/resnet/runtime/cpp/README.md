English | [简体中文](README_cn.md)

# ResNet18 C++ runtime (S-series)

The S-series ResNet18 native runtime: the `hbDNNInferV2` inference flow,
image preprocessing, NV12 tensor creation and Top-K output, built on the
shared `utils/c_utils` sources.

<a id="overview"></a>
## C++ inference

Use this directory for c++ inference.

<a id="directory"></a>
## Directory structure

```text
cpp/
├── inc/
│   ├── classify.hpp  # Resnet18 model class and owned stage-data types
│   └── cli.hpp       # CLI options and helpers
├── src/
│   ├── classify.cpp  # runtime lifecycle, preprocess/infer/postprocess
│   ├── cli.cpp       # argument parsing, defaults, image/label loading, printing
│   └── main.cpp      # entry point: parse options, predict, print
├── CMakeLists.txt    # build (C++17, explicit RDK_TARGET board selection)
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
| X5 | not-supported |

The CMake file reads `/sys/class/boardinfo/soc_name` and defines the matching
SoC macro; an unreadable identity file is an error, not a fallback.

<a id="dependencies"></a>
## Dependencies

On the board image: CMake and a C++17 compiler; OpenCV development
headers/libraries; `fmt` development libraries; Horizon DNN headers under
`/usr/hobot/include` and libraries under `/usr/hobot/lib` (`hbDNN`,
`hbucp`). Utility implementations come from the shared
`utils/c_utils` files referenced by the CMake target. The launcher does
not install system packages, modify the SDK, or download a model.

<a id="build"></a>
## Build

The launcher builds automatically; to build manually (cwd: repository
root):

```bash
# success: build directory contains the resnet18 binary
cmake -S samples/vision/resnet/runtime/cpp \
  -B samples/vision/resnet/runtime/cpp/build
cmake --build samples/vision/resnet/runtime/cpp/build --parallel
```

An alternative build directory selects the same target:

```bash
cmake -S samples/vision/resnet/runtime/cpp \
  -B /tmp/resnet18-alt-build
cmake --build /tmp/resnet18-alt-build --parallel
```

<a id="run"></a>
## Run

Prerequisite: `bash samples/vision/resnet/model/download.sh s100` (or the
S600 artifact). From any working directory:

```bash
# input: model/s100 artifact, bundled zebra_cls.jpg, S ImageNet labels
# output: Top-K lines on stdout — success: exit 0
bash samples/vision/resnet/runtime/cpp/run.sh
```

For S600, select the artifact explicitly and use a separate build
directory when the same checkout serves both boards:

```bash
MODEL_PATH="$PWD/samples/vision/resnet/model/s600/resnet18_224x224_nv12.hbm" \
BUILD_DIR="$PWD/samples/vision/resnet/runtime/cpp/build-s600" \
bash samples/vision/resnet/runtime/cpp/run.sh
```

The launcher checks the model, image, and label files first, then invokes
CMake, builds `resnet18`, and passes absolute paths so the command works
from any directory.

<a id="parameters"></a>
## Parameters

Options of the `resnet18` binary, matching the Python runtime's
kebab-case names (the launcher overrides the first three with absolute
sample paths):

| Option | Default | Meaning |
| --- | --- | --- |
| `--model-path` | SoC-dependent: `/opt/hobot/model/s100/basic/resnet18_224x224_nv12.hbm` (S100) or `/opt/hobot/model/s600/basic/resnet18_224x224_nv12.hbm` (S600) | HBM model path |
| `--test-img` | `../../../test_data/zebra_cls.jpg` (relative to the process working directory) | BGR test image |
| `--label-file` | repository S ImageNet labels path | one label per line |
| `--top-k` | `5` | number of printed classes |
| `--help` / `-h` | — | print usage |

Example override through the launcher:

```bash
bash samples/vision/resnet/runtime/cpp/run.sh \
  --model-path /opt/hobot/model/s100/basic/resnet18_224x224_nv12.hbm \
  --test-img /tmp/zebra_cls.jpg \
  --label-file /tmp/imagenet_classes.names \
  --top-k 5
```

<a id="interface-lifecycle"></a>
## Interface and lifecycle

`main.cpp` parses the options, constructs `Resnet18 model(model_path)` —
the constructor loads the HBM pack, reads the tensor metadata and
allocates the reusable tensor buffers — then calls
`model.predict(image, top_k)` and prints the returned classes. All DNN
and UCP types stay inside `src/classify.cpp` (private `Impl`), so
`inc/classify.hpp` depends only on OpenCV and the standard library.

The model exposes the preprocessing, inference and postprocessing stages
separately, each returning data owned by the caller:

- `Resnet18Prepared preprocess(const cv::Mat&)` — letterbox resize to the
  model input resolution and BGR→NV12 conversion into owned Y/UV planes;
- `Resnet18Raw infer(const Resnet18Prepared&)` — upload of the planes
  into the model input tensors (row-stride aware), one
  `hbDNNInferV2` BPU task, copy of the F32 output into an owned logits
  vector that survives later inferences;
- `std::vector<Classification> postprocess(const Resnet18Raw&, int top_k)`
  — stable softmax and Top-K selection;
- `predict` composes the three stages in that order.

Errors surface as C++ exceptions (SDK error descriptions included); the
entry point prints them and exits with status 2. Resources are released
by RAII on every path, including partial initialization failures. There
is no background thread; the process performs one synchronous inference.

<a id="results-interpretation"></a>
## Results interpretation

The binary prints the Top-K classes using the linewise ImageNet label
file, one result per line with class id, score, and label; the exit code
is 0 on success. Record the board identity, artifact reference, complete
build/run command, and Top-K output for each S100 or S600 evaluation.
