# ResNet18 C++ runtime (S-series)

The consolidated S-series ResNet18 native runtime: the audited S18
`hbDNNInferV2` flow, image preprocessing, NV12 tensor creation, and Top-K
output code kept as source in the canonical sample. The old S18 C++
directory remains a thin CMake compatibility configure path; it does not
maintain a second copy.

<a id="supported-boards"></a>
## Supported boards

| Board | Status |
| --- | --- |
| S100 | supported-verified (built and run 2026-09-17; Top-5 equal to the source baseline) |
| S600 | supported-not-run (same source and SoC detection; board access unavailable) |
| X5 | not-supported (no X5 C++ source exists in the audited baseline) |

The CMake file reads `/sys/class/boardinfo/soc_name` and defines the SoC
macro used by the original source; an unreadable identity file is an error,
not a fallback.

<a id="dependencies"></a>
## Dependencies

On the board image: CMake and a C++17 compiler; OpenCV development
headers/libraries; `gflags` and `fmt` development libraries; Horizon DNN
headers under `/usr/hobot/include` and libraries under `/usr/hobot/lib`
(`hbDNN`, `hbucp`). The source utility implementations come from the
existing `platforms/s/utils/c_utils` files referenced by the canonical
CMake target. The launcher does not install system packages, modify the
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

The compatibility path selects the same canonical target:

```bash
cmake -S platforms/s/samples/vision/resnet18/runtime/cpp \
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
utility implementations are the existing `platforms/s/utils/c_utils`
sources. There is no background thread; the process performs one
synchronous inference.

<a id="results-interpretation"></a>
## Results interpretation

The binary prints the Top-K classes using the linewise ImageNet label
file, one result per line with class id, score, and label; the exit code
is 0 on success. Record the board identity, artifact reference, complete
build/run command, and Top-K output for every native evaluation. S600
connectivity or an unavailable artifact is `not-run`, not a successful
S100 substitute.
