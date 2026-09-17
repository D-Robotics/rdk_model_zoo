# C++ runtime

This is the consolidated S-series ResNet18 native runtime. It keeps the
audited S18 `hbDNNInferV2` flow, image preprocessing, NV12 tensor creation, and
Top-K output code in the canonical sample. The old S18 C++ directory is a thin
CMake compatibility configure path and adds this target; it does not maintain
a second copy of the implementation.

## Board prerequisites

Build and run on an RDK S100 or S600 board with the matching board image. The
host/board environment must already provide:

* CMake and a C++17 compiler;
* OpenCV development headers and libraries;
* `gflags` and `fmt` development libraries;
* Horizon DNN headers under `/usr/hobot/include` and libraries under
  `/usr/hobot/lib`, including `hbDNN` and `hbucp`.

The CMake file reads `/sys/class/boardinfo/soc_name` and defines the SoC macro
used by the original source. The launcher does not install system packages,
modify the SDK, or download a model.

## Prepare and run

Prepare the S100 artifact explicitly from the repository root, then run:

```bash
bash samples/vision/resnet/model/download.sh s100
bash samples/vision/resnet/runtime/cpp/run.sh
```

The launcher checks the model, bundled `zebra_cls.jpg`, and
`platforms/s/datasets/imagenet/imagenet_classes.names`, configures CMake,
builds the `resnet18` binary, and passes those paths to it. For S600, use the
S600 artifact and a separate build directory:

```bash
bash samples/vision/resnet/model/download.sh s600
MODEL_PATH="$PWD/samples/vision/resnet/model/s600/resnet18_224x224_nv12.hbm" \
BUILD_DIR="$PWD/samples/vision/resnet/runtime/cpp/build-s600" \
bash samples/vision/resnet/runtime/cpp/run.sh
```

The script accepts native flag overrides such as:

```bash
bash samples/vision/resnet/runtime/cpp/run.sh \
  --model_path /opt/hobot/model/s100/basic/resnet18_224x224_nv12.hbm \
  --test_img /tmp/zebra_cls.jpg \
  --label_file /tmp/imagenet_classes.names \
  --top_k 5
```

Overrides are resolved before file checks and passed through to the binary.
Use a path that exists on the board; no download is implied by a missing file.

## Native flow

`main.cpp` creates `Resnet18`, loads the HBM, converts the BGR image through
`pre_process`, invokes `hbDNNInferV2`, and decodes the output with
`post_process`. The C++ runtime receives the S-series Y and UV input tensors and
prints the configured Top-K classes using the linewise ImageNet label file.
The source utility implementations are the existing files under
`platforms/s/utils/c_utils`; their include and source paths are referenced by
the canonical CMake target.

The default binary flags retain the old names and defaults:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model_path` | board S100 `/opt/hobot/model/s100/basic/...` or S600 equivalent | HBM model path |
| `--test_img` | `../../../test_data/zebra_cls.jpg` from the build directory | BGR test image |
| `--label_file` | repository S ImageNet labels | one label per line |
| `--top_k` | `5` | number of printed classes |

The launcher supplies absolute sample paths so its command works from any
working directory. The binary defaults remain for callers that invoke it
directly from the historical build layout.

## Compatibility build and troubleshooting

To verify that the old configure path still selects the canonical target:

```bash
cmake -S platforms/s/samples/vision/resnet18/runtime/cpp \
  -B /tmp/resnet18-legacy-build
cmake --build /tmp/resnet18-legacy-build --parallel
```

The old `runtime/cpp/run.sh` delegates to the canonical launcher while
retaining its old model/image/label locations. If CMake cannot read the SoC,
the board identity file is unavailable. If headers or libraries are missing,
install the board image/toolchain prerequisites through the normal platform
process and rerun; this sample does not perform that installation. If the
binary reports a model or input failure, compare the artifact target, tensor
protocol, model metadata, image path, and label path before changing source.

Record the board identity, artifact reference, complete build command, and
Top-K output for every native evaluation. S600 connectivity or an unavailable
artifact is `not-run`, not a successful S100 result.
