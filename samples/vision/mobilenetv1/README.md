English | [简体中文](README_cn.md)

# MobileNetV1 image classification

MobileNetV1 uses depthwise separable convolution for lightweight image classification.

Sources: [tensorflow/models MobileNetV1](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) · [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

The sample ships one Python runtime for all targets. The `MobileNetV1Classifier` class runs a `preprocess → infer → postprocess` flow chained by `predict`: it resolves one exact artifact reference from the platform release manifest for the detected board, verifies the board identity, loads `hbm_runtime` lazily, and returns a typed Top-K result
([runtime/python/README.md](runtime/python/README.md)).

### Algorithm background

MobileNetV1 targets efficient image classification on embedded and mobile
devices. Its efficiency comes from the depthwise separable convolution,
which factorizes a standard convolution into a per-channel depthwise
filter and a 1×1 pointwise projection that combines the channel outputs
([paper](https://arxiv.org/abs/1704.04861),
[tensorflow/models MobileNetV1](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)).

Feature summary:

- **Depthwise separable convolution**: decomposes a standard convolution into depthwise convolution and a 1×1 pointwise convolution.
- **Lightweight design**: reduces computation and parameter count for embedded deployment.
- **Classification output**: Top-K class IDs and confidence scores for ImageNet-1k labels.

![Depthwise and pointwise convolution](./test_data/depthwise&pointwise.png)

*Depthwise separable convolution: each input channel is filtered by its own D_K×D_K depthwise
kernel; the following 1×1 pointwise convolution mixes the per-channel
results.*

<a id="directory"></a>
## Directory structure

```text
mobilenetv1/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python inference
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
| x5 | mobilenetv1 | python | supported |
| s100 | mobilenetv1 | python | supported |
| s600 | mobilenetv1 | python | supported |
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
python3 -m venv .venv-mobilenetv1
source .venv-mobilenetv1/bin/activate
python3 -m pip install -r samples/vision/mobilenetv1/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

One complete path on an X5 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row x5:mobilenetv1:mobilenetv1_224x224_nv12.bin)
#    output: samples/vision/mobilenetv1/model/mobilenetv1_224x224_nv12.bin
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/mobilenetv1/model/download.sh x5

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
python3 samples/vision/mobilenetv1/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv1:mobilenetv1_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv1/model/mobilenetv1_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv1/test_data/bulbul.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

For S100/S600 use the matching `s:` reference (see `--list-models`) and the
same root `datasets/imagenet/` labels. Full commands:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; Pass `--img-save-path` to save a visualization; otherwise results are printed to stdout. On X5 with the bundled `bulbul.JPEG` the Top-1 matches the
image subject (a bulbul (bird)); on S100/S600 with `zebra_cls.jpg` the
Top-5 includes `zebra`. Select a target and variant listed in the [Support matrix](#support-matrix), prepare that exact manifest artifact with the model downloader, and run the sample on the matching board.

<a id="performance"></a>
## Performance data

Published MobileNetV1 performance on `RDK X5` (x5-v1.1.3):

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV1 | 224x224 | 1000 | 4.2 | 71.7% | 65.4% | 0.58 | 2800+ |


![Inference result](./test_data/inference.png)

*Reference inference result from the X5 release: the
bundled [bulbul.JPEG](test_data/bulbul.JPEG) ranks `bulbul` first,
followed by junco/snowbird, robin, chickadee, and water ouzel. This is the source-reported X5 runtime example.*

<a id="entry-points"></a>
## Entry points

- Model preparation: [model/README.md](model/README.md)
- Python runtime: [runtime/python/README.md](runtime/python/README.md)
- Conversion: [conversion/README.md](conversion/README.md)
- Evaluation: [evaluator/README.md](evaluator/README.md)

<a id="license"></a>
## License

Sample code follows the repository top-level LICENSE (Apache-2.0). The
source model is the upstream MobileNetV1 distribution; upstream model/weights
licensing is governed by that distribution (see the reference-implementation
link above). Published artifacts follow the platform release manifests; the
manifests carry no separate license field, and no additional license is
claimed here.
