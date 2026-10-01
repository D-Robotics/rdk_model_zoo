# MobileOne image classification

<a id="overview"></a>

## Overview

MobileOne is a lightweight CNN backbone designed for low-latency deployment on edge devices. The model uses structural re-parameterization to keep training-time expressiveness while simplifying the inference-time structure.

- **Paper**: [MobileOne: An Improved One millisecond Mobile Backbone](http://arxiv.org/abs/2206.04040)
- **Reference Implementation**: [apple/ml-mobileone](https://github.com/apple/ml-mobileone)

The source README's feature highlights:

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
the branches folded into a single 3×3 / 1×1 convolution (right). Restored
from the X5 source README (rdk_x5 @ac115717197920355fc390bb04299b20e6436864);
the train/inference split explains why the deployed artifacts are the
already-reparameterized INT8 s0–s4 variants at 224×224 NV12 (see [Support
matrix](#support-matrix)).*

One BGR image produces ImageNet-1k Top-K class IDs, scores and optional labels. The unified Python task delegates preprocessing, inference and postprocessing through the existing shared classification implementation. Labels, drawing and file output belong to the CLI.

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Python | C++ |
| --- | --- | --- | --- |
| x5 | s0 | supported-not-run | not-supported |
| x5 | s1 | supported-not-run | not-supported |
| x5 | s2 | supported-not-run | not-supported |
| x5 | s3 | supported-not-run | not-supported |
| x5 | s4 | supported-not-run | not-supported |
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

`supported-not-run` means an implementation and published artifact exist, but unified board validation has not run. No S-series artifacts or C++ implementations are provided. [Host validation records](../../../docs/releases/unified-migration/2026-09-22-b4-classification-review.md) do not establish board verification.

Source: rdk_x5 @ac115717197920355fc390bb04299b20e6436864. No C++ runtime is delivered for this sample. The lowercase CLI IDs map to exact published filenames; letter casing in filenames is preserved.

<a id="prerequisites"></a>
## Prerequisites

Use a full repository checkout. On X5 use the matching board image and its `hbm_runtime`; install host dependencies in a virtual environment as below. SciPy is used only by the preserved-source comparison tests, not unified inference.

Locally tested host environment: Python 3.14.7, NumPy 2.5.3, OpenCV 4.14.0, PyYAML 6.0.3 and SciPy 1.18.1. This is a host regression environment, not a qualified board dependency set. X5 4GB/8GB validation is planned; exact board image, Python and SDK versions and minimum RAM remain unverified. Allow disk space for the checkout, selected model and outputs; a minimum capacity has not been measured. Native inference needs no OE toolchain; conversion prerequisites are documented under conversion.

```bash
# cwd: repository root
python3 -m venv .venv-mobileone
source .venv-mobileone/bin/activate
python3 -m pip install -r samples/vision/mobileone/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

Run on X5 from the repository root. Download exits 0 and prints an observed digest; inference exits 0 and prints five results. No automatic download occurs during inference.

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

Default variant `s0` preserves the source entrypoint. Choose `s1`, `s2`, `s3`, `s4` explicitly. Source softmax scores produce a stable Top-K, with exact ties ordered by ascending class ID. `tiger_beetle.JPEG` is a functional input, not dataset accuracy evidence; unified board results are not available yet. No file is saved unless `--img-save-path` is given.

For reference, the X5 source README (rdk_x5
@ac115717197920355fc390bb04299b20e6436864, legacy Python entrypoint)
illustrated its run with the screenshot below: the legacy `result.jpg`
drawing overlaid the top-5 ranks on the image, with rank 1 being class
300 (tiger beetle) on the bundled `tiger_beetle.JPEG`. This is a
historical screenshot from the source delivery, not a run of the current
entrypoint in this repository.

![Historical inference screenshot from the X5 source README: tiger
beetle test image with the legacy top-5 overlay, rank 1 class 300 (tiger
beetle)](./test_data/inference.png)

<a id="performance"></a>
## Performance data

Historical source records, not remeasured here. Full timing conditions and all columns are retained in [evaluation](evaluator/README.md#reference-results). Do not compare single-thread latency with multi-thread FPS as reciprocal quantities.

<a id="directory"></a>
## Directory

`model/`: artifacts and download; `runtime/python/`: native CLI, task and runner; `conversion/`: 5 unchanged PTQ YAMLs; `evaluator/`: checks and historical benchmarks; `test_data/`: `tiger_beetle.JPEG` input and accompanying resources; `tests/`: host regressions.

<a id="entry-points"></a>
## Entry points

[Model](model/README.md) · [Python](runtime/python/README.md) · [Conversion](conversion/README.md) · [Evaluation](evaluator/README.md)

The old `platforms/x5/samples/vision/mobileone` entry remains the original source implementation for baseline comparisons. It has not become a forwarding shim. New integrations use this sample; internal legacy imports are not promised compatible.

<a id="license"></a>
## License

Source Python headers retain Apache-2.0 provenance. Conversion YAMLs retain their original proprietary notices verbatim; the repository license is not a grant overriding those notices or upstream weights licenses. Check the applicable notices before redistributing conversion material or weights.
