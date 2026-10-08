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
├── inc/  # Files for inc
├── src/  # Files for src
├── CMakeLists.txt  # Source or data file
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── run.sh  # Run the sample
```

<a id="supported-boards"></a>
## Supported boards

| Board | Status |
| --- | --- |
| S100 | supported |
| S600 | supported |
| X5 | not-supported |

The CMake file reads `/sys/class/boardinfo/soc_name` and defines the SoC
macro used by the original source; an unreadable identity file is an error,
not a fallback.

<a id="dependencies"></a>
## Dependencies

On the board image: CMake and a C++17 compiler; OpenCV development
headers/libraries; `gflags` and `fmt` development libraries; Horizon DNN
headers under `/usr/hobot/include` and libraries under `/usr/hobot/lib`
(`hbDNN`, `hbucp`). The source utility implementations come from the
existing `utils/c_utils` files referenced by the CMake target. The launcher does not install system packages, modify the
SDK, or download a model.

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
  -B /tmp/resnet18-legacy-build
cmake --build /tmp/resnet18-legacy-build --parallel
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

Native gflags of the `resnet18` binary (the launcher overrides the first
three with absolute sample paths):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model_path` | SoC-dependent: `/opt/hobot/model/s100/basic/resnet18_224x224_nv12.hbm` (S100) or `/opt/hobot/model/s600/basic/resnet18_224x224_nv12.hbm` (S600) | HBM model path |
| `--test_img` | `../../../test_data/zebra_cls.jpg` (relative to the historical build layout) | BGR test image |
| `--label_file` | repository S ImageNet labels path | one label per line |
| `--top_k` | `5` | number of printed classes |

Example override through the launcher:

```bash
bash samples/vision/resnet/runtime/cpp/run.sh \
  --model_path /opt/hobot/model/s100/basic/resnet18_224x224_nv12.hbm \
  --test_img /tmp/zebra_cls.jpg \
  --label_file /tmp/imagenet_classes.names \
  --top_k 5
```

<a id="interface-lifecycle"></a>
## Interface and lifecycle

`main.cpp` creates the `Resnet18` model object, loads the HBM and extracts
tensor metadata, converts the BGR image through the model's preprocessing
(NV12 Y/UV tensor creation), invokes `hbDNNInferV2` on the S-series input
tensors, decodes the F32 output with the Top-K postprocess, prints the
configured Top-K classes, and releases the DNN resources at scope exit.
The heavy work happens after construction, not in the constructor; the
utility implementations are the existing `utils/c_utils`
sources. There is no background thread; the process performs one
synchronous inference.

<a id="results-interpretation"></a>
## Results interpretation

The binary prints the Top-K classes using the linewise ImageNet label
file, one result per line with class id, score, and label; the exit code
is 0 on success. Record the board identity, artifact reference, complete
build/run command, and Top-K output for each S100 or S600 evaluation.
