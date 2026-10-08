English | [简体中文](README_cn.md)

# MobileNetV2 image classification (C++, S-series)

This C++ flow runs the quantised MobileNetV2 HBM model on an S100 or S600
BPU and prints Top-K class labels with confidence scores. It uses the
`hbDNNInferV2` API; the Python runtime provides the X5 path.

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

On S100 and S600, the launcher reads `/sys/class/boardinfo/soc_name` and
`/sys/class/boardinfo/board_type` to select the matching runtime path. The
registered identity names are listed in `docs/release/platforms.json`.
`SOC_NAME_FILE` and `BOARD_TYPE_FILE` select alternate identity files for
local runs; on a board, use the system identity files.

<a id="dependencies"></a>
## Dependencies

CMake, a C++17 compiler, OpenCV development packages, `libgflags-dev`,
and the Horizon DNN headers/libraries of the board image. Install the required compiler and board SDK development packages before building:

```bash
# cwd: on the board — success: apt reports the packages installed
sudo apt update && sudo apt install -y libgflags-dev
```

<a id="build"></a>
## Build

Manual build (cwd: `samples/vision/mobilenetv2/runtime/cpp`; success:
`build/` contains the `mobilenetv2` binary):

```bash
mkdir -p build && cd build && cmake .. && make -j"$(nproc)"
```

`CMakeLists.txt` detects the SoC at configure time via
`/sys/class/boardinfo/soc_name` and defines `SOC_S100`/`SOC_S600`. On
small-RAM boards, build with `make -j1` or `BUILD_JOBS=1 bash run.sh`.

<a id="run"></a>
## Run

One-command form (cwd: anywhere; prerequisite: the artifact prepared per
[model/README.md](../../model/README.md); success: exit 0 and a printed
Top-5 list):

```bash
bash samples/vision/mobilenetv2/runtime/cpp/run.sh
```

The launcher builds into `runtime/cpp/build/` and executes the binary
with the prepared model, `test_data/zebra_cls.jpg`, and
`test_data/imagenet1000_labels.txt`. Environment overrides: `MODEL_PATH`,
`TEST_IMAGE`, `LABEL_FILE`, `TOP_K`, `BUILD_DIR`, `BUILD_JOBS`. Prepare the model artifact with the command in [model/README.md](../../model/README.md) before running the launcher.

<a id="parameters"></a>
## Parameters

| Parameter | Description | Default (from the launcher) |
| --- | --- | --- |
| `--model_path` | Path to the `.hbm` artifact | `model/<soc>/mobilenetv2_224x224_nv12.hbm` relative to the sample |
| `--test_img` | Test image path | `test_data/zebra_cls.jpg` relative to the sample |
| `--label_file` | Label file path | `test_data/imagenet1000_labels.txt` relative to the sample |
| `--top_k` | Number of Top-K results to print | `5` |

The binary's compiled-in defaults point at `/opt/hobot/model/...`; the
launcher always passes explicit paths, so the system-model location is
used only if you pass it yourself.

<a id="interface-lifecycle"></a>
## Interface and lifecycle

`mobilenetv2::init` loads the model, allocates tensors, and reads the
layout metadata; `pre_process`, `infer`, and `post_process` are free
functions passing tensors by reference (declaration in
`inc/mobilenetv2.hpp`). Doxygen comments live in the source; the
repository-level API reference build is described in
`docs/source_reference/README.md`.

<a id="results-interpretation"></a>
## Results interpretation

On success the Top-K lines print `TOP-n: label=..., prob=...` with the
label file's names and the artifact's post-softmax scores. The rdk_s
@s-v1.1.2 record with `zebra_cls.jpg` on S100:

```text
TOP-1: label=zebra, prob=0.992246
TOP-2: label=tiger, Panthera tigris, prob=0.00404656
TOP-3: label=hartebeest, prob=0.00133707
TOP-4: label=tiger cat, prob=0.000722661
TOP-5: label=impala, Aepyceros melampus, prob=0.000539704
```

A correct run reproduces this ordering within score noise; with the
bundled zebra image the TOP-1 is `zebra`. Scores that are all zero or NaN
indicate a wrong artifact/input pairing, not a tuning problem.
