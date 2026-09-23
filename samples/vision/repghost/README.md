# RepGhost image classification

<a id="overview"></a>
## Overview

RepGhost is a lightweight CNN family designed to improve hardware efficiency by replacing explicit feature reuse in feature space with re-parameterized reuse in weight space. It avoids costly `Concat` operations while keeping strong classification performance.

- **Paper**: [RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization](https://arxiv.org/abs/2211.06088)
- **Reference Implementation**: [ChengpengChen/RepGhost](https://github.com/ChengpengChen/RepGhost)

One BGR image produces ImageNet-1k Top-K class IDs, scores and optional labels. The unified Python task delegates preprocessing, inference and postprocessing through the existing shared classification implementation. Labels, drawing and file output belong to the CLI.

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Python | C++ |
| --- | --- | --- | --- |
| x5 | 100 | supported-not-run | not-supported |
| x5 | 111 | supported-not-run | not-supported |
| x5 | 130 | supported-not-run | not-supported |
| x5 | 150 | supported-not-run | not-supported |
| x5 | 200 | supported-not-run | not-supported |
| s100 | 100 | not-supported | not-supported |
| s100 | 111 | not-supported | not-supported |
| s100 | 130 | not-supported | not-supported |
| s100 | 150 | not-supported | not-supported |
| s100 | 200 | not-supported | not-supported |
| s100p | 100 | not-supported | not-supported |
| s100p | 111 | not-supported | not-supported |
| s100p | 130 | not-supported | not-supported |
| s100p | 150 | not-supported | not-supported |
| s100p | 200 | not-supported | not-supported |
| s600 | 100 | not-supported | not-supported |
| s600 | 111 | not-supported | not-supported |
| s600 | 130 | not-supported | not-supported |
| s600 | 150 | not-supported | not-supported |
| s600 | 200 | not-supported | not-supported |

`supported-not-run` means an implementation and published artifact exist, but unified board validation has not run. No S-series artifacts or C++ implementations are provided. [Host validation records](../../../docs/releases/unified-migration/2026-09-22-b4-classification-review.md) do not establish board verification.

Source: `rdk_x5 @ac115717197920355fc390bb04299b20e6436864`. Neither source delivers RepGhost C++. No board result is claimed for this migration.

<a id="prerequisites"></a>
## Prerequisites

Use a full repository checkout. On X5 use the matching board image and its `hbm_runtime`; install host dependencies in a virtual environment as below. SciPy is used only by the preserved-source comparison tests, not unified inference.

Locally tested host environment: Python 3.14.7, NumPy 2.5.3, OpenCV 4.14.0, PyYAML 6.0.3 and SciPy 1.18.1. This is a host regression environment, not a qualified board dependency set. X5 4GB/8GB validation is planned; exact board image, Python and SDK versions and minimum RAM remain unverified. Allow disk space for the checkout, selected model and outputs; a minimum capacity has not been measured. Native inference needs no OE toolchain; conversion prerequisites are documented under conversion.

```bash
# cwd: repository root
python3 -m venv .venv-repghost
source .venv-repghost/bin/activate
python3 -m pip install -r samples/vision/repghost/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

Run on X5 from the repository root. Download exits 0 and prints an observed digest; inference exits 0 and prints five results. No automatic download occurs during inference.

```bash
# cwd: repository root
bash samples/vision/repghost/model/download.sh x5 100
python3 samples/vision/repghost/runtime/python/main.py \
  --target x5 --variant 100 \
  --test-img samples/vision/repghost/test_data/ibex.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="expected-results"></a>
## Expected results

The default is variant `100`, preserving the source entrypoint. `111`, `130`, `150`, `200` must be selected explicitly. Top-K scores use the source softmax policy; exact ties use stable ascending class-ID order. The ibex image is a functional input, not a dataset accuracy test; no current-board reference output exists yet. No image is saved unless `--img-save-path` is supplied.

<a id="performance"></a>
## Performance data

Historical source records, not remeasured here. Full timing conditions and all columns are retained in [evaluation](evaluator/README.md#reference-results). Do not compare single-thread latency with multi-thread FPS as reciprocal quantities.

<a id="directory"></a>
## Directory

`model/`: artifacts and download; `runtime/python/`: native CLI, task and runner; `conversion/`: five unchanged PTQ YAMLs; `evaluator/`: checks and historical benchmarks; `test_data/`: `ibex.JPEG` input and accompanying resources; `tests/`: host regressions.

<a id="entry-points"></a>
## Entry points

[Model](model/README.md) · [Python](runtime/python/README.md) · [Conversion](conversion/README.md) · [Evaluation](evaluator/README.md)

The old `platforms/x5/samples/vision/repghost` entry remains the original source implementation for baseline comparisons. It has not become a forwarding shim. New integrations use this sample; internal legacy imports are not promised compatible.

<a id="license"></a>
## License

Source Python headers retain Apache-2.0 provenance. Conversion YAMLs retain their original proprietary notices verbatim; the repository license is not a grant overriding those notices or upstream weights licenses. Check the applicable notices before redistributing conversion material or weights.
