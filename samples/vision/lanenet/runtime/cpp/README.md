English | [简体中文](README_cn.md)

# LaneNet C++ runtime

This entry is a DNN/UCP implementation producing an embedding map and binary lane labels. The `segment` module owns the model lifecycle, tensor contract and the preprocessing, inference and decoding stages; the CLI module owns rendering and report IO. It does **not** cluster embeddings into lane instances or fit curves.

<a id="overview"></a>
## C++ inference

Use this directory for c++ inference.

<a id="directory"></a>
## Directory structure

```text
cpp/
├── inc/
│   ├── segment.hpp  # LaneNet model class, tensor contract and owned stage-data types
│   └── cli.hpp      # CLI options, NPY/JSON serialization and artifact writer
├── src/
│   ├── segment.cpp  # runtime lifecycle, tensor binding, preprocess/infer/postprocess
│   ├── cli.cpp      # argument parsing, image loading, visualization, report writing
│   └── main.cpp     # entry point: parse options, predict, save results
├── CMakeLists.txt   # build (C++17, explicit RDK_TARGET S100 gate)
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── launcher.py  # Python script
└── run.sh  # Run the sample
```

<a id="supported-boards"></a>
## Supported boards

| Target | Published contract | Runtime support |
| --- | --- | --- |
| S100 | `s:lanenet:s100/lanenet256x512.hbm` | supported |
| S100P / S600 / X5 | No asset for this sample | Rejected; no S100 fallback |

Both the launcher and native owner check board identity before SDK use. The launcher checks before building. `--dry-run` and `--list-models` require neither a board nor SDK. `auto` resolves the only published contract, S100; actual runs still require the board's S100 identity and SDK.

<a id="dependencies"></a>
## Dependencies

Prepare a matching S100 DNN/UCP development environment, CMake >=3.16, a C++17 compiler, OpenCV development libraries (`core`, `imgproc`, `imgcodecs`) and threads. The launcher uses Python 3 and the repository's shared manifest support; it does not install dependencies or download a model. The native executable does not use Python at inference time.

CMake locates `hobot/dnn/hb_dnn.h`, `hobot/hb_ucp.h`, `dnn` and `hbucp`. Nonstandard installations can set `DNN_INCLUDE_DIR`, `UCP_INCLUDE_DIR`, `DNN_LIBRARY`, `UCP_LIBRARY` and `OpenCV_DIR` in a manual CMake invocation. Use headers and libraries from the same SDK installation. The [tests](../../tests) directory provides failure-injection fixtures with fake SDK headers.

<a id="build"></a>
## Build

Commands below run from the repository root. First inspect the selection without building or loading the SDK:

```bash
python3 samples/vision/lanenet/runtime/cpp/launcher.py --target s100 --dry-run --build
```

Prepare the model explicitly using [the model instructions](../../model/README.md). On an S100 with the development dependencies installed, this command checks identity, model and output paths, builds into `runtime/cpp/build/s100`, then runs:

```bash
bash samples/vision/lanenet/runtime/cpp/run.sh --target s100 --build --output outputs/lanenet_cpp_first
```

To configure a different SDK location manually, configure this directory with CMake and provide the discovery variables above, then build target `lanenet`. That build produces the executable; run it per the run section to execute inference, with the complete build performed against the real SDK/OpenCV development environment.

<a id="run"></a>
## Run

After a successful build, omit `--build` to reuse the sample binary:

```bash
bash samples/vision/lanenet/runtime/cpp/run.sh --target s100 --output outputs/lanenet_cpp_next
```

For an explicitly supplied executable and model, bind the file to the published contract. The manifest has no publisher checksum; the launcher records the local model digest in `launch-report.json` to identify the bytes used for the run.

```bash
python3 samples/vision/lanenet/runtime/cpp/launcher.py --target s100 --asset-id s:lanenet:s100/lanenet256x512.hbm --model-path /data/models/lanenet256x512.hbm --binary /data/bin/lanenet --test-img samples/vision/lanenet/test_data/lane.jpg --output outputs/lanenet_cpp_external
```

Use a new output directory for every run. Return code 0 requires a native `report.json`; errors return 2, or preserve a nonzero native return code. If inference fails before creating the result directory, logs are emitted to the terminal and there is no saved launch report. Preserve terminal output when diagnosing such failures.

<a id="parameters"></a>
## Parameters

These defaults belong to the Python launcher, also used by `run.sh`:

| Option | Default | Meaning |
| --- | --- | --- |
| `--target` | `s100` | `auto` or S100 contract; other listed targets are rejected |
| `--asset-id` | `null` | Inferred published asset; required with an external model path |
| `--model-path` | `null` | Resolves to sample `model/s100/lanenet256x512.hbm` |
| `--test-img` | `samples/vision/lanenet/test_data/lane.jpg` | BGR image decoded by OpenCV |
| `--output` | `outputs/lanenet_cpp` | New result directory |
| `--instance-save-path` | `null` | Optional additional embedding display |
| `--binary-save-path` | `null` | Optional additional binary display |
| `--binary` | `null` | Resolves to `runtime/cpp/build/s100/lanenet` |
| `--build` | `false` | Explicit build; mutually exclusive with `--binary` |
| `--list-models` | `false` | List published manifest identity without execution |
| `--dry-run` | `false` | Print resolved command without execution |

Additional image paths must be new, distinct and must not overwrite result files or launch logs. Relative paths passed to the launcher are resolved before it starts the binary from the repository root. The shell helper first changes to the repository root.

The direct binary requires `--model-path` and `--test-img`. Its `--target` and `--output` defaults are `s100` and `outputs/lanenet_cpp`; extra image paths are empty by default. It accepts the underscore aliases (`--model_path`, `--test_img`, `--instance_save_path`, `--binary_save_path`) and `--key=value`. It has no manifest selection, automatic build, download or digest recording; use the launcher for run provenance. UCP scheduling uses the default priority and the ANY core.

<a id="interface-lifecycle"></a>
## Interface and lifecycle

`main.cpp` parses the options, constructs `LaneNet model(model_path)` — the constructor verifies the board's S100 identity before any SDK call, loads the HBM pack, validates the tensor contract and allocates the reusable tensor buffers — then calls `model.predict(image)` and hands the result to the artifact writer. All DNN and UCP types stay inside `src/segment.cpp` (private `Impl`), so `inc/segment.hpp` depends only on OpenCV and the standard library.

[LaneNet](inc/segment.hpp) exposes the three stages separately, each returning data owned by the caller:

| Stage | Input | Output / contract |
| --- | --- | --- |
| `preprocess` | Nonempty `CV_8UC3` BGR image | Owned contiguous float32 NCHW `[1,3,256,512]`; BGR→RGB, INTER_AREA resize, /255 and ImageNet normalization |
| `infer` | Prepared float vector | Owned raw tensors with actual shape, dtype, byte strides and allocation size |
| `postprocess` | Raw tensor vector | `LaneResult`: owned raw outputs plus float32 CHW `[3,256,512]` embedding and uint8 `[256,512]` labels (values 0 or 1) |
| `predict` | BGR image | Composes the three stages; no file writes or visualization |

Native output roles require exactly one float32 `[1,3,256,512]` embedding and exactly one int64 `[1,1,256,512]` or `[1,256,512]` binary tensor. Roles are resolved by unique shape and type, not assumed indices. Ambiguity is an error. Other observed numeric outputs are retained by index without inventing names or semantics; native output names are not queried. The [Python entry](../python/README.md) binds the two required names exposed by its SDK.

Tensor copying honors every byte stride, including width padding, and rejects overlapping or out-of-capacity layouts. SDK call results are checked on every path — including task submission, wait and release — and a finished task is released exactly once; scoped cleanup covers partial initialization, allocation, submit and wait failures. Errors surface as C++ exceptions; the entry point prints them and exits with status 2. Returned tensors own their bytes and remain valid after the inference task is released.

<a id="results-interpretation"></a>
## Results and interpretation

| File | Meaning |
| --- | --- |
| `raw_output_N.npy` | Every observed output, packed without padding, preserving exact dtype and shape; int64 is not converted through floating point |
| `embedding.npy` | Raw float32 `[3,256,512]` embedding, not clipped or clustered |
| `binary.npy` | Validated 0/1 uint8 `[256,512]` labels |
| `instance_pred.png` | Display of clipped embedding channels; colors are not lane IDs |
| `binary_pred.png` | Binary labels multiplied by 255 |
| `report.json` | Actual tensor metadata, output-role indices, model name and processing boundaries |
| `launch-report.json` | Launcher command, UTC interval, return code and observed model/input/binary/report digests |
| `native.stdout.log`, `native.stderr.log` | Complete captured native streams when a result directory exists |

All images remain on the 256×512 model grid. No original-size restoration, clustering, tracking, curve fitting, accuracy measurement or latency measurement is performed. Embedding display clips to [0,1], multiplies by 255 and rounds ties to even. Use raw embeddings for numerical comparisons and the images for visualization. They also verify int64 values above float64's exact integer range. Full native OpenCV/SDK builds and board inference run in the target environment. For missing dependencies inspect CMake discovery, for identity rejection check the physical board, and for metadata rejection retain the actual metadata rather than renaming tensors or forcing a target.
