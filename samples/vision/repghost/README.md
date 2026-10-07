# RepGhost image classification

<a id="overview"></a>

## Overview

RepGhost is a lightweight CNN family designed to improve hardware efficiency by replacing explicit feature reuse in feature space with re-parameterized reuse in weight space. It avoids costly `Concat` operations while keeping strong classification performance.

- **Paper**: [RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization](https://arxiv.org/abs/2211.06088)
- **Reference Implementation**: [ChengpengChen/RepGhost](https://github.com/ChengpengChen/RepGhost)

Feature highlights:

- **Structural re-parameterization** — converts training-time complex
  branches into efficient inference-time structures.
- **Implicit feature reuse** — moves GhostNet-style feature reuse from
  feature space (`Concat`) to weight space, avoiding costly memory
  copies.
- **Hardware efficiency** — reduces memory-copy overhead and improves
  deployment efficiency on edge devices.
- **Variant scaling** — published variants from `100` to `200`.

![RepGhost bottleneck compared with the Ghost bottleneck: training-time
add branches fused for inference](./test_data/RepGhost_architecture.png)

*Figure (upstream paper Fig. 4): (a) Ghost bottleneck with its explicit
`Concat` feature reuse; (b) RG-bneck at training — reuse moves to weight
space via `add` branches; (c) RG-bneck at inference — the branches are
fused away. it depicts the upstream
architecture, while the deployed artifacts are the INT8-quantized
100–200 variants at 224×224 NV12 (see [Support
matrix](#support-matrix)).*

One BGR image produces ImageNet-1k Top-K class IDs, scores and optional
labels. The `RepGhostClassifier` class runs a `preprocess → infer → postprocess` flow
chained by `predict` (labels, drawing and file output belong to the CLI
layer; see [runtime/python/README.md](runtime/python/README.md)).

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Python | C++ |
| --- | --- | --- | --- |
| x5 | 100 | supported | not-supported |
| x5 | 111 | supported | not-supported |
| x5 | 130 | supported | not-supported |
| x5 | 150 | supported | not-supported |
| x5 | 200 | supported | not-supported |
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

Select a supported target and runtime from the support matrix.

<a id="prerequisites"></a>
## Prerequisites

Use a full repository checkout. On X5 use the matching board image and its `hbm_runtime`; install host dependencies in a virtual environment as below. Host-side comparison tools use SciPy; board inference uses the dependencies in the matching runtime image.

Use a full repository checkout. On X5 use the matching board image and
its `hbm_runtime`; install the host dependencies in a virtual environment
as below. Host-side comparison tools use SciPy; board inference uses the dependencies in the matching runtime image. Native inference needs no OE toolchain; conversion
prerequisites are documented under [conversion](conversion/README.md).
```bash
# cwd: repository root
python3 -m venv .venv-repghost
source .venv-repghost/bin/activate
python3 -m pip install -r samples/vision/repghost/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

From the repository root, prepare the X5 artifact with the model downloader, then run inference with the bundled image and the selected artifact reference.

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

The default variant is `100`; `111`, `130`, `150`, `200` must be selected
explicitly. Softmax scores produce a stable Top-K; exact ties use stable
ascending class-ID order. For dataset accuracy, use the validation-set procedure in the evaluator guide. No image is saved unless `--img-save-path` is
supplied.

The screenshot below shows a reference run from the X5 release: the
demo overlay draws the top-5 ranks onto the bundled test image, with
rank 1 being class 350 (ibex, Capra ibex).

![Reference inference result on X5: ibex test image with the top-5 overlay, rank 1 class 350 (ibex, Capra ibex)](./test_data/inference.png)

<a id="performance"></a>
## Performance data

Published performance records; full timing conditions and all columns
are listed under [evaluation](evaluator/README.md#reference-results).
Single-thread latency and multi-thread FPS are measured under different
concurrency and are not reciprocal quantities. Compare latency and FPS using
the same thread count, concurrent submission mode and BPU utilization.

<a id="directory"></a>
## Directory

`model/`: artifacts and download; `runtime/python/`: native CLI, task and runner; `conversion/`: five PTQ YAMLs; `evaluator/`: functional checks and published benchmarks; `test_data/`: `ibex.JPEG` input and accompanying resources; `tests/`: host unittest suite.

<a id="entry-points"></a>
## Entry points

[Model](model/README.md) · [Python](runtime/python/README.md) · [Conversion](conversion/README.md) · [Evaluation](evaluator/README.md)


<a id="license"></a>
## License

Source Python headers retain Apache-2.0 provenance. Conversion YAMLs retain their original proprietary notices verbatim; Before redistribution, follow the original conversion YAML notices and applicable upstream weight license terms.
