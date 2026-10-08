English | [简体中文](README_cn.md)

# MobileOne image classification

MobileOne folds training branches into a simple convolutional network for inference.

[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

MobileOne is a lightweight CNN backbone designed for low-latency deployment on edge devices. The model uses structural re-parameterization to keep training-time expressiveness while simplifying the inference-time structure.

- **Paper**: [MobileOne: An Improved One millisecond Mobile Backbone](http://arxiv.org/abs/2206.04040)
- **Reference Implementation**: [apple/ml-mobileone](https://github.com/apple/ml-mobileone)

Feature highlights:

- **Structural re-parameterization** — fuses multi-branch training blocks
  (k parallel conv branches plus branch-wise BN and an identity branch,
  with ReLU or SE-ReLU) into a single inference-friendly conv per block.
- **Low-latency backbone** — targets mobile and embedded deployment with
  strong throughput.
- **Variant scaling** — published variants from `S0` to `S4`, with the
  over-parameterization factor `k` tuned per variant.
- **Classification output** — Top-K class IDs and confidence scores for
  ImageNet-1k labels.

![MobileOne block: train-time multi-branch structure reparameterized
into the inference-time plain conv](./test_data/MobileOne_architecture.png)

*Figure (upstream paper Fig. 3): the MobileOne block has two structures —
train time with reparameterizable branches (left), inference time with
the branches folded into a single 3×3 / 1×1 convolution (right). The
train/inference split explains why the deployed artifacts are the
already-reparameterized INT8 s0–s4 variants at 224×224 NV12 (see [Support
matrix](#support-matrix)).*

One BGR image produces ImageNet-1k Top-K class IDs, scores and optional
labels. The `MobileOneClassifier` class runs a `preprocess → infer → postprocess` flow
chained by `predict` (labels, drawing and file output belong to the CLI
layer; see [runtime/python/README.md](runtime/python/README.md)).

<a id="directory"></a>
## Directory structure

```text
mobileone/
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
| x5 | s0 | supported | not-supported |
| x5 | s1 | supported | not-supported |
| x5 | s2 | supported | not-supported |
| x5 | s3 | supported | not-supported |
| x5 | s4 | supported | not-supported |
| s100 | s0 | not-supported | not-supported |
| s100 | s1 | not-supported | not-supported |
| s100 | s2 | not-supported | not-supported |
| s100 | s3 | not-supported | not-supported |
| s100 | s4 | not-supported | not-supported |
| s100p | s0 | not-supported | not-supported |
| s100p | s1 | not-supported | not-supported |
| s100p | s2 | not-supported | not-supported |
| s100p | s3 | not-supported | not-supported |
| s100p | s4 | not-supported | not-supported |
| s600 | s0 | not-supported | not-supported |
| s600 | s1 | not-supported | not-supported |
| s600 | s2 | not-supported | not-supported |
| s600 | s3 | not-supported | not-supported |
| s600 | s4 | not-supported | not-supported |

Use the Python CLI with a lowercase variant ID; it resolves to the exact case-sensitive published filename.

<a id="prerequisites"></a>
## Prerequisites

Use a full repository checkout. On X5 use the matching board image and its `hbm_runtime`; install host dependencies in a virtual environment as below. Host-side comparison tools use SciPy; board inference uses the dependencies in the matching runtime image.

Use a full repository checkout. On X5 use the matching board image and
its `hbm_runtime`; install the host dependencies in a virtual environment
as below. Host-side comparison tools use SciPy; board inference uses the dependencies in the matching runtime image. Native inference needs no OE toolchain; conversion
prerequisites are documented under [conversion](conversion/README.md).
```bash
# cwd: repository root
python3 -m venv .venv-mobileone
source .venv-mobileone/bin/activate
python3 -m pip install -r samples/vision/mobileone/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

From the repository root, prepare the X5 artifact with the model downloader, then run inference with the bundled image and the selected artifact reference.

```bash
# cwd: repository root
bash samples/vision/mobileone/model/download.sh x5 s0
python3 samples/vision/mobileone/runtime/python/main.py \
  --target x5 --variant s0 \
  --test-img samples/vision/mobileone/test_data/tiger_beetle.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="expected-results"></a>
## Expected results

The default variant is `s0`; the other variants are chosen explicitly. Softmax scores produce a stable Top-K, with exact ties ordered by ascending class ID. `tiger_beetle.JPEG` is the bundled functional input; use the evaluator guide for dataset accuracy. Pass `--img-save-path` to save an image; otherwise results are printed to stdout.

The screenshot below shows a reference run from the X5 release: the
demo overlay draws the top-5 ranks onto the bundled test image, with
rank 1 being class 300 (tiger beetle).

![Reference inference result on X5: tiger beetle test image with the top-5 overlay, rank 1 class 300 (tiger beetle)](./test_data/inference.png)

<a id="performance"></a>
## Performance data

Published performance records; full timing conditions and all columns
are listed under [evaluation](evaluator/README.md#reference-results).
Single-thread latency and multi-thread FPS are measured under different
concurrency and are not reciprocal quantities. Compare latency and FPS using
the same thread count, concurrent submission mode and BPU utilization.

<a id="entry-points"></a>
## Entry points

[Model](model/README.md) · [Python](runtime/python/README.md) · [Conversion](conversion/README.md) · [Evaluation](evaluator/README.md)

<a id="license"></a>
## License

Source Python headers retain Apache-2.0 provenance. Conversion YAMLs retain their original proprietary notices verbatim; Before redistribution, follow the original conversion YAML notices and applicable upstream weight license terms.
