English | [简体中文](README_cn.md)

# MobileNetV2 image classification (C++, S-series)

This C++ flow runs the quantised MobileNetV2 HBM model (variant `100` or `140`)
on an S100, S100P or S600 BPU and prints Top-K class labels with confidence
scores. It uses the
`hbDNNInferV2` API; the Python runtime provides the X5 path.

<a id="overview"></a>
## C++ inference

Use this directory for c++ inference.

<a id="directory"></a>
## Directory structure

```text
cpp/
├── inc/
│   ├── classify.hpp  # MobileNetV2 model class and owned stage-data types
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

On S100, S100P and S600, the launcher reads `/sys/class/boardinfo/soc_name` and
`/sys/class/boardinfo/board_type` to select the board's own artifact (`nashe`,
`nashm` or `nashp`); S100P is soc_name `s100` with board_type `s100p` or
`rdk s100p`. The registered identity names are listed in
`docs/release/platforms.json`; unknown boards are rejected.
`SOC_NAME_FILE` and `BOARD_TYPE_FILE` select alternate identity files for
local runs; on a board, use the system identity files.

<a id="dependencies"></a>
## Dependencies

CMake, a C++17 compiler, OpenCV development packages, `fmt` development
libraries, and the Horizon DNN headers/libraries of the board image. Use the SDK development packages
provided by your board image and perform the full SDK build in that
environment; the launcher never calls apt.

<a id="build"></a>
## Build

Manual build (cwd: `samples/vision/mobilenetv2/runtime/cpp`; success:
`build/` contains the `mobilenetv2` binary):

```bash
mkdir -p build && cd build && cmake .. && make -j"$(nproc)"
```

`CMakeLists.txt` selects the board at configure time: natively on a board
it reads `/sys/class/boardinfo/soc_name` (`-DRDK_TARGET=auto`, the
default) and defines `SOC_S100`/`SOC_S100P`/`SOC_S600`; cross compilation must pass an
explicit `-DRDK_TARGET=s100|s100p|s600` (auto is rejected while cross-compiling,
and unsupported targets fail the configure). On small-RAM boards, build
with `make -j1` or `BUILD_JOBS=1 bash run.sh`.

<a id="run"></a>
## Run

One-command form (cwd: anywhere; prerequisite: the artifact prepared per
[model/README.md](../../model/README.md); success: exit 0 and a printed
Top-5 list):

```bash
bash samples/vision/mobilenetv2/runtime/cpp/run.sh
```

The launcher builds into `runtime/cpp/build/` and executes the binary
with the prepared model of `VARIANT` (default `100`; `VARIANT=140 bash run.sh`
selects the other width), `test_data/zebra_cls.jpg`, and
`test_data/imagenet1000_labels.txt`. Environment overrides: `VARIANT`,
`MODEL_PATH`, `TEST_IMAGE`, `LABEL_FILE`, `TOP_K`, `BUILD_DIR`, `BUILD_JOBS`. Prepare the model artifact with the command in [model/README.md](../../model/README.md) before running the launcher.

<a id="parameters"></a>
## Parameters

| Parameter | Description | Default (from the launcher) |
| --- | --- | --- |
| `--model-path` | Path to the `.hbm` artifact | `model/<soc>/mobilenetv2_<variant>_<march>_224x224_nv12.hbm` relative to the sample |
| `--test-img` | Test image path | `test_data/zebra_cls.jpg` relative to the sample |
| `--label-file` | Label file path | `test_data/imagenet1000_labels.txt` relative to the sample |
| `--top-k` | Number of Top-K results to print | `5` |
| `--resize-shorter` | Shorter edge before the center crop, `int(224 / crop_pct)` | `256` (both variants) |

Option names are kebab-case to match the Python runtime, and both the
`--flag value` and `--flag=value` spellings are accepted. The binary's
compiled-in model default is the variant-100 artifact under the sample's
`model/<board>/` directory; the launcher always passes explicit paths.

<a id="interface-lifecycle"></a>
## Interface and lifecycle

`main.cpp` parses the options, constructs `MobileNetV2 model(model_path, resize_shorter)`
— the constructor loads the HBM pack, reads and validates the tensor
metadata and allocates the reusable tensor buffers — then calls
`model.predict(image, top_k)` and prints the returned classes. All DNN
and UCP types stay inside `src/classify.cpp` (private `Impl`), so
`inc/classify.hpp` depends only on OpenCV and the standard library.

The model exposes the preprocessing, inference and postprocessing stages
separately, each returning data owned by the caller:

- `MobileNetV2Prepared preprocess(const cv::Mat&)` — antialiased bicubic
  shorter-edge resize to `resize_shorter` (a bit-exact port of Pillow's,
  shared with `utils/tools/mobilenet/cpp/geometry.hpp`), center crop to the
  model input and BGR→NV12 conversion into owned Y/UV planes, the same
  geometry as the Python runtime;
- `MobileNetV2Raw infer(const MobileNetV2Prepared&)` — upload of the
  planes into the model input tensors (row-stride aware), one
  `hbDNNInferV2` BPU task, copy of the F32 output into an owned
  logits vector that survives later inferences;
- `std::vector<Classification> postprocess(const MobileNetV2Raw&, int top_k)`
  — numerically stable softmax over the logits and Top-K selection;
- `predict` composes the three stages in that order.

Errors surface as C++ exceptions (SDK error descriptions included); the
entry point prints them and exits with status 2. Resources are released
by RAII on every path, including partial initialization failures. There
is no background thread; the process performs one synchronous inference.

<a id="results-interpretation"></a>
## Results interpretation

On success the Top-K lines print `TOP-n: label=..., prob=...` with the
label file's names and softmax scores. Output of `bash run.sh` (variant `100`)
with `zebra_cls.jpg` on S100 with the published build:

```text
TOP-1: label=zebra, prob=0.808423
TOP-2: label=tiger, Panthera tigris, prob=0.00355534
TOP-3: label=hartebeest, prob=0.00321221
TOP-4: label=tiger cat, prob=0.00207029
TOP-5: label=ostrich, Struthio camelus, prob=0.00169806
```

S100P prints the same values; S600 prints `zebra` first with 0.797359. With
`VARIANT=140` the TOP-1 is `zebra` at 0.942 on all three boards. The scores
equal those of the Python runtime on the same board, because both use the same
preprocessing. Scores that are all zero or NaN
indicate a wrong artifact/input pairing, not a tuning problem.
