English | [简体中文](./README_cn.md)

# MobileNetV2 image classification (C++, S-series)

This C++ flow runs the quantised MobileNetV2 HBM model on the S-series
BPU and prints Top-K class labels with confidence scores. It is the
audited S-series `hbDNNInferV2` implementation kept from rdk_s @380e1a2
(`src/` and `inc/` are verbatim); the X5 source branch delivered Python
only, so this flow declares S-series scope explicitly.

<a id="supported-boards"></a>
## Supported boards

S100 and S600 only. The launcher reads the board identity once from
`/sys/class/boardinfo/soc_name` and rejects s100p and unknown boards with
an explicit error — the legacy source launcher silently fell back to the
s100 artifact for them, which this sample does not keep.

<a id="dependencies"></a>
## Dependencies

CMake, a C++17 compiler, OpenCV development packages, `libgflags-dev`,
and the Horizon DNN headers/libraries of the board image. Install them
explicitly (the launcher never invokes apt):

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
`/sys/class/boardinfo/soc_name` and defines `SOC_S100`/`SOC_S600`; it
stays verbatim from the source branch. On small-RAM boards (observed on
S100, 2026-09-21) full-parallel compiles can be OOM-killed — build with
`make -j1` or `BUILD_JOBS=1 bash run.sh` there.

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
`TEST_IMAGE`, `LABEL_FILE`, `TOP_K`, `BUILD_DIR`, `BUILD_JOBS`. It never
downloads a model: a missing artifact is an explicit error naming the
download command.

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

`mobilenetv2::init()` loads the model, allocates tensors, and reads the
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

A correct unified run reproduces this ordering within score noise; the
B1 board comparison against this baseline is pending, so this flow's
status row stays not-run until it is recorded. Scores that are all zero
or NaN indicate a wrong artifact/input pairing, not a tuning problem.
