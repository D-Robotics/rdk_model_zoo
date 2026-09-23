# RepVGG image classification

<a id="overview"></a>
## Overview

RepVGG is a VGG-style convolutional network family that uses structural re-parameterization. During training it can use multi-branch structures, while during deployment it is converted into a plain stack of `3x3` convolution and ReLU layers for efficient inference.

- **Paper**: [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
- **Reference Implementation**: [DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)

One BGR image produces ImageNet-1k Top-K class IDs, scores and optional labels. The unified Python task delegates preprocessing, inference and postprocessing through the existing shared classification implementation. Labels, drawing and file output belong to the CLI.

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Python | C++ |
| --- | --- | --- | --- |
| x5 | a0 | supported-not-run | not-supported |
| x5 | a1 | supported-not-run | not-supported |
| x5 | a2 | supported-not-run | not-supported |
| x5 | b0 | supported-not-run | not-supported |
| x5 | b1g2 | supported-not-run | not-supported |
| x5 | b1g4 | supported-not-run | not-supported |
| s100 | a0 | not-supported | not-supported |
| s100 | a1 | not-supported | not-supported |
| s100 | a2 | not-supported | not-supported |
| s100 | b0 | not-supported | not-supported |
| s100 | b1g2 | not-supported | not-supported |
| s100 | b1g4 | not-supported | not-supported |
| s100p | a0 | not-supported | not-supported |
| s100p | a1 | not-supported | not-supported |
| s100p | a2 | not-supported | not-supported |
| s100p | b0 | not-supported | not-supported |
| s100p | b1g2 | not-supported | not-supported |
| s100p | b1g4 | not-supported | not-supported |
| s600 | a0 | not-supported | not-supported |
| s600 | a1 | not-supported | not-supported |
| s600 | a2 | not-supported | not-supported |
| s600 | b0 | not-supported | not-supported |
| s600 | b1g2 | not-supported | not-supported |
| s600 | b1g4 | not-supported | not-supported |

`supported-not-run` means an implementation and published artifact exist, but unified board validation has not run. No S-series artifacts or C++ implementations are provided. [Host validation records](../../../docs/releases/unified-migration/2026-09-22-b4-classification-review.md) do not establish board verification.

Source: rdk_x5 @ac115717197920355fc390bb04299b20e6436864. No C++ runtime is delivered for this sample. The lowercase CLI IDs map to exact published filenames; letter casing in filenames is preserved.

<a id="prerequisites"></a>
## Prerequisites

Use a full repository checkout. On X5 use the matching board image and its `hbm_runtime`; install host dependencies in a virtual environment as below. SciPy is used only by the preserved-source comparison tests, not unified inference.

Locally tested host environment: Python 3.14.7, NumPy 2.5.3, OpenCV 4.14.0, PyYAML 6.0.3 and SciPy 1.18.1. This is a host regression environment, not a qualified board dependency set. X5 4GB/8GB validation is planned; exact board image, Python and SDK versions and minimum RAM remain unverified. Allow disk space for the checkout, selected model and outputs; a minimum capacity has not been measured. Native inference needs no OE toolchain; conversion prerequisites are documented under conversion.

```bash
# cwd: repository root
python3 -m venv .venv-repvgg
source .venv-repvgg/bin/activate
python3 -m pip install -r samples/vision/repvgg/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

Run on X5 from the repository root. Download exits 0 and prints an observed digest; inference exits 0 and prints five results. No automatic download occurs during inference.

```bash
# cwd: repository root
bash samples/vision/repvgg/model/download.sh x5 a0
python3 samples/vision/repvgg/runtime/python/main.py \
  --target x5 --variant a0 \
  --test-img samples/vision/repvgg/test_data/gooze.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="expected-results"></a>
## Expected results

Default variant `a0` preserves the source entrypoint. Choose `a1`, `a2`, `b0`, `b1g2`, `b1g4` explicitly. Source softmax scores produce a stable Top-K, with exact ties ordered by ascending class ID. `gooze.JPEG` is a functional input, not dataset accuracy evidence; unified board results are not available yet. No file is saved unless `--img-save-path` is given.

<a id="performance"></a>
## Performance data

Historical source records, not remeasured here. Full timing conditions and all columns are retained in [evaluation](evaluator/README.md#reference-results). Do not compare single-thread latency with multi-thread FPS as reciprocal quantities.

<a id="directory"></a>
## Directory

`model/`: artifacts and download; `runtime/python/`: native CLI, task and runner; `conversion/`: 6 unchanged PTQ YAMLs; `evaluator/`: checks and historical benchmarks; `test_data/`: `gooze.JPEG` input and accompanying resources; `tests/`: host regressions.

<a id="entry-points"></a>
## Entry points

[Model](model/README.md) · [Python](runtime/python/README.md) · [Conversion](conversion/README.md) · [Evaluation](evaluator/README.md)

The old `platforms/x5/samples/vision/repvgg` entry remains the original source implementation for baseline comparisons. It has not become a forwarding shim. New integrations use this sample; internal legacy imports are not promised compatible.

<a id="license"></a>
## License

Source Python headers retain Apache-2.0 provenance. Conversion YAMLs retain their original proprietary notices verbatim; the repository license is not a grant overriding those notices or upstream weights licenses. Check the applicable notices before redistributing conversion material or weights.
