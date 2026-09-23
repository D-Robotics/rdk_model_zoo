# RepViT image classification

<a id="overview"></a>
## Overview

RepViT revisits lightweight mobile CNN design from a ViT perspective. It keeps a pure CNN deployment structure while borrowing lightweight ViT design ideas, and improves inference efficiency through structural reparameterization.

- **Paper**: [RepViT: Revisiting Mobile CNN From ViT Perspective](http://arxiv.org/abs/2307.09283)
- **Reference Implementation**: [THU-MIG/RepViT](https://github.com/THU-MIG/RepViT)

One BGR image produces ImageNet-1k Top-K class IDs, scores and optional labels. The unified Python task delegates preprocessing, inference and postprocessing through the existing shared classification implementation. Labels, drawing and file output belong to the CLI.

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Python | C++ |
| --- | --- | --- | --- |
| x5 | m0_9 | supported-not-run | not-supported |
| x5 | m1_0 | supported-not-run | not-supported |
| x5 | m1_1 | supported-not-run | not-supported |
| s100 | m0_9 | not-supported | not-supported |
| s100 | m1_0 | not-supported | not-supported |
| s100 | m1_1 | not-supported | not-supported |
| s100p | m0_9 | not-supported | not-supported |
| s100p | m1_0 | not-supported | not-supported |
| s100p | m1_1 | not-supported | not-supported |
| s600 | m0_9 | not-supported | not-supported |
| s600 | m1_0 | not-supported | not-supported |
| s600 | m1_1 | not-supported | not-supported |

`supported-not-run` means an implementation and published artifact exist, but unified board validation has not run. No S-series artifacts or C++ implementations are provided. [Host validation records](../../../docs/releases/unified-migration/2026-09-22-b4-classification-review.md) do not establish board verification.

Source: rdk_x5 @ac115717197920355fc390bb04299b20e6436864. No C++ runtime is delivered for this sample. The lowercase CLI IDs map to exact published filenames; letter casing in filenames is preserved.

<a id="prerequisites"></a>
## Prerequisites

Use a full repository checkout. On X5 use the matching board image and its `hbm_runtime`; install host dependencies in a virtual environment as below. SciPy is used only by the preserved-source comparison tests, not unified inference.

Locally tested host environment: Python 3.14.7, NumPy 2.5.3, OpenCV 4.14.0, PyYAML 6.0.3 and SciPy 1.18.1. This is a host regression environment, not a qualified board dependency set. X5 4GB/8GB validation is planned; exact board image, Python and SDK versions and minimum RAM remain unverified. Allow disk space for the checkout, selected model and outputs; a minimum capacity has not been measured. Native inference needs no OE toolchain; conversion prerequisites are documented under conversion.

```bash
# cwd: repository root
python3 -m venv .venv-repvit
source .venv-repvit/bin/activate
python3 -m pip install -r samples/vision/repvit/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

Run on X5 from the repository root. Download exits 0 and prints an observed digest; inference exits 0 and prints five results. No automatic download occurs during inference.

```bash
# cwd: repository root
bash samples/vision/repvit/model/download.sh x5 m0_9
python3 samples/vision/repvit/runtime/python/main.py \
  --target x5 --variant m0_9 \
  --test-img samples/vision/repvit/test_data/yurt.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="expected-results"></a>
## Expected results

Default variant `m0_9` preserves the source entrypoint. Choose `m1_0`, `m1_1` explicitly. Source softmax scores produce a stable Top-K, with exact ties ordered by ascending class ID. `yurt.JPEG` is a functional input, not dataset accuracy evidence; unified board results are not available yet. No file is saved unless `--img-save-path` is given.

<a id="performance"></a>
## Performance data

Historical source records, not remeasured here. Full timing conditions and all columns are retained in [evaluation](evaluator/README.md#reference-results). Do not compare single-thread latency with multi-thread FPS as reciprocal quantities.

<a id="directory"></a>
## Directory

`model/`: artifacts and download; `runtime/python/`: native CLI, task and runner; `conversion/`: 3 unchanged PTQ YAMLs; `evaluator/`: checks and historical benchmarks; `test_data/`: `yurt.JPEG` input and accompanying resources; `tests/`: host regressions.

<a id="entry-points"></a>
## Entry points

[Model](model/README.md) · [Python](runtime/python/README.md) · [Conversion](conversion/README.md) · [Evaluation](evaluator/README.md)

The old `platforms/x5/samples/vision/repvit` entry remains the original source implementation for baseline comparisons. It has not become a forwarding shim. New integrations use this sample; internal legacy imports are not promised compatible.

<a id="license"></a>
## License

Source Python headers retain Apache-2.0 provenance. Conversion YAMLs retain their original proprietary notices verbatim; the repository license is not a grant overriding those notices or upstream weights licenses. Check the applicable notices before redistributing conversion material or weights.
