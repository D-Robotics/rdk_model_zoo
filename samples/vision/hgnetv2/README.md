# HGNetV2 image classification

HGNetV2 is a convolutional backbone for image classification.

[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

HGNetV2 is a convolutional backbone for vision tasks; this sample exposes
b0–b4 ImageNet-1k classification models. HGNetV2 is a next-generation CNN
backbone designed for a strong accuracy/latency balance, succeeding the
original HGNet, and performing well in classification, detection and
segmentation.

[PP-HGNetV2](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/en/models/ImageNet1k/PP-HGNetV2.md)

Feature highlights:

- **Aggregating multiple receptive fields** — the HG-Block combines
  multi-scale features from shallow to deep layers, which is friendly to
  small-object detection and recognition.
- **Improved stem module** — the input stem stacks more 2×2 convolution
  kernels to learn rich local features while using smaller channel
  numbers, boosting performance on high-resolution tasks.
- **Learnable downsampling (LDS)** — an adaptive downsampling layer
  preserves more useful spatial details while reducing computational
  redundancy.

One BGR image produces ImageNet-1k Top-K class IDs, scores and optional
labels. The `HGNetV2Classifier` class runs a `preprocess → infer →
postprocess` flow chained by `predict` (labels, drawing and file output
belong to the CLI layer; see
[runtime/python/README.md](runtime/python/README.md)).

<a id="directory"></a>
## Directory structure

```text
hgnetv2/
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

| Target | Variant | Python | C++ |
| --- | --- | --- | --- |
| x5 | b0 | supported | not-supported |
| x5 | b1 | supported | not-supported |
| x5 | b2 | supported | not-supported |
| x5 | b3 | supported | not-supported |
| x5 | b4 | supported | not-supported |
| s100 | b0 | not-supported | not-supported |
| s100 | b1 | not-supported | not-supported |
| s100 | b2 | not-supported | not-supported |
| s100 | b3 | not-supported | not-supported |
| s100 | b4 | not-supported | not-supported |
| s100p | b0 | not-supported | not-supported |
| s100p | b1 | not-supported | not-supported |
| s100p | b2 | not-supported | not-supported |
| s100p | b3 | not-supported | not-supported |
| s100p | b4 | not-supported | not-supported |
| s600 | b0 | not-supported | not-supported |
| s600 | b1 | not-supported | not-supported |
| s600 | b2 | not-supported | not-supported |
| s600 | b3 | not-supported | not-supported |
| s600 | b4 | not-supported | not-supported |

Select a supported target and runtime from the support matrix. The lowercase CLI IDs
map to exact published filenames; letter casing in filenames is
preserved.

<a id="prerequisites"></a>
## Prerequisites

Use a full repository checkout. On X5 use the matching board image and
its `hbm_runtime`; install the host dependencies in a virtual environment
as below. Host-side comparison tools use SciPy; board inference uses the dependencies in the matching runtime image. Native inference needs no OE toolchain; conversion
prerequisites are documented under [conversion](conversion/README.md).

```bash
# cwd: repository root
python3 -m venv .venv-hgnetv2
source .venv-hgnetv2/bin/activate
python3 -m pip install -r samples/vision/hgnetv2/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

From the repository root, prepare the X5 artifact with the model downloader, then run inference with the bundled image and the selected artifact reference.

```bash
# cwd: repository root
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/runtime/python/main.py \
  --target x5 --variant b0 \
  --test-img samples/vision/hgnetv2/test_data/sandbar.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="expected-results"></a>
## Expected results

The default variant is `b0`; choose `b1`, `b2`, `b3`, `b4` explicitly.
Softmax scores produce a stable Top-K, with exact ties ordered by
ascending class ID. `sandbar.JPEG` is the bundled functional input; use the evaluator guide for dataset accuracy. Pass `--img-save-path` to save an image; otherwise results are printed to stdout.

The screenshot below shows a reference run from the X5 release: the demo
overlay draws the top-5 ranks onto the bundled `sandbar.JPEG`, with rank 1
being class 977 (sandbar, sand bar).

![Reference inference result on X5: sandbar test image with the top-5
overlay, rank 1 class 977 (sandbar, sand bar)](./test_data/result.jpg)

<a id="performance"></a>
## Performance data

Published performance records; full timing conditions and all columns
are listed under [evaluation](evaluator/README.md#reference-results).
Single-thread latency and multi-thread FPS are measured under different
concurrency and are not reciprocal quantities. Compare latency and FPS using
the same thread count, concurrent submission mode and BPU utilization.

<a id="entry-points"></a>
## Entry points

[Model](model/README.md) · [Python](runtime/python/README.md) ·
[Conversion](conversion/README.md) · [Evaluation](evaluator/README.md)

<a id="license"></a>
## License

Python code follows Apache-2.0. Follow the original conversion notices and applicable upstream model and weight license terms. Review the upstream model and weight license terms before redistribution.
