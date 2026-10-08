English | [简体中文](README_cn.md)

# MobileNetV2 image classification

MobileNetV2 uses inverted residual blocks and linear bottlenecks for lightweight image classification.

Sources: [timm/models/mobilenetv2](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv2.py) · [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

The sample ships one Python runtime for all targets plus one S-series C++
runtime. The `MobileNetV2Classifier` class runs a `preprocess → infer →
postprocess` flow chained by `predict`: it resolves one exact artifact
reference from the platform release manifest for the detected board,
verifies the board identity, loads `hbm_runtime` lazily, and returns a
typed Top-K result
([runtime/python/README.md](runtime/python/README.md)). The C++ flow is
the S-series `hbDNNInferV2` implementation
([runtime/cpp/README.md](runtime/cpp/README.md)).

### Algorithm background

MobileNetV2 introduces inverted residual blocks with linear bottlenecks:
each block expands the channels with a 1×1 convolution, applies a 3×3
depthwise convolution, and projects back through a linear (non-ReLU) 1×1
bottleneck; stride-2 blocks drop the shortcut. The linear bottleneck
keeps information that ReLU would discard in low-dimensional spaces
([paper](https://arxiv.org/abs/1801.04381),
[timm/models/mobilenetv2](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv2.py)).

Feature summary:

- **Inverted residuals**: expand channels before the depthwise convolution and project back through a linear bottleneck.
- **Depthwise separable convolution**: reduces computation compared with standard convolution.
- **Classification output**: Top-K class IDs and confidence scores for ImageNet-1k labels.

![MobileNetV2 architecture](./test_data/mobilenetv2_architecture.png)

*Inverted residual blocks: the stride-1 block (left) keeps the additive shortcut; the
stride-2 block (right) downsamples without it, and only the final 1×1
projection is linear.*

The source tree also shipped the paper's block-evolution figure
(`test_data/seperated_conv.png`):

![Evolution of separable convolution blocks](./test_data/seperated_conv.png)

*Figure 2 of the MobileNetV2 paper: from a regular convolution (a) to a
separable block (b), a separable block with linear bottleneck (c), and a
bottleneck with expansion layer (d); hatched layers carry no
non-linearity.*

<a id="directory"></a>
## Directory structure

```text
mobilenetv2/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python and native inference implementations
├── test_data/  # Example inputs
├── tests/  # Automated tests
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── requirements-host.txt  # Python dependencies
```

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Language | Status |
| --- | --- | --- | --- |
| x5 | mobilenetv2 | python | supported |
| s100 | mobilenetv2 | python | supported |
| s600 | mobilenetv2 | python | supported |
| s100 | mobilenetv2 | cpp | supported |
| s600 | mobilenetv2 | cpp | supported |
| s100p | any | python, cpp | not-supported (no s100p asset row in the release manifest; selection is an explicit error, no fallback) |

<a id="prerequisites"></a>
## Prerequisites

Board Python execution needs the board image's matching `hbm_runtime`,
NumPy, and OpenCV-Python; the SDK is imported only when a model executes
(`--help`, `--list-models`, `--dry-run`, and host tests need no SDK).
Manifest reading also needs PyYAML. On a development host, install the
user-space dependencies from `requirements-host.txt`:

```bash
# cwd: repository root — success: "host dependencies: ok"
python3 -m venv .venv-mobilenetv2
source .venv-mobilenetv2/bin/activate
python3 -m pip install -r samples/vision/mobilenetv2/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

The C++ build needs CMake, a C++17 compiler, OpenCV and gflags
development packages, and the Horizon DNN headers/libraries — see
[runtime/cpp/README.md](runtime/cpp/README.md).

<a id="quickstart"></a>
## Quick start

One complete path on an X5 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row x5:mobilenetv2:mobilenetv2_224x224_nv12.bin)
#    output: samples/vision/mobilenetv2/model/mobilenetv2_224x224_nv12.bin
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/mobilenetv2/model/download.sh x5

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
python3 samples/vision/mobilenetv2/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv2:mobilenetv2_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv2/model/mobilenetv2_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv2/test_data/Scottish_deerhound.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

For S100/S600 use the matching `s:` reference (see `--list-models`) and the
same root `datasets/imagenet/` labels. Full commands:
[runtime/python/README.md](runtime/python/README.md).
For the C++ flow use `bash samples/vision/mobilenetv2/runtime/cpp/run.sh`.

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; Pass `--img-save-path` to save a visualization; otherwise results are printed to stdout. On X5 with the bundled `Scottish_deerhound.JPEG` the Top-1 matches the
image subject (a Scottish deerhound (dog)); on S100/S600 with `zebra_cls.jpg` the
Top-5 includes `zebra`. Select a target and variant listed in the [Support matrix](#support-matrix), prepare that exact manifest artifact with the model downloader, and run the sample on the matching board.

<a id="performance"></a>
## Performance data

Published MobileNetV2 performance on `RDK X5` (x5-v1.1.3):

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV2 | 224x224 | 1000 | 3.4 | 72.0% | 68.17% | 1.42 | 1152.07 |


![Inference result](./test_data/inference.png)

*Reference inference result from the X5 release: the
bundled [Scottish_deerhound.JPEG](test_data/Scottish_deerhound.JPEG)
ranks `Scottish deerhound` first, followed by Irish wolfhound, lynx/
catamount, standard schnauzer, and timber wolf. This is the source-reported X5 runtime example.*

<a id="entry-points"></a>
## Entry points

- Model preparation: [model/README.md](model/README.md)
- Python runtime: [runtime/python/README.md](runtime/python/README.md)
- C++ runtime (S100/S600): [runtime/cpp/README.md](runtime/cpp/README.md)
- Conversion: [conversion/README.md](conversion/README.md)
- Evaluation: [evaluator/README.md](evaluator/README.md)

<a id="license"></a>
## License

Sample code follows the repository top-level LICENSE (Apache-2.0). The
source model is the upstream MobileNetV2 distribution; upstream model/weights
licensing is governed by that distribution (see the reference-implementation
link above). Published artifacts follow the platform release manifests; the
manifests carry no separate license field, and no additional license is
claimed here.
