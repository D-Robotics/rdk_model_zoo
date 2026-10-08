English | [简体中文](README_cn.md)

# GoogLeNet image classification

GoogLeNet uses Inception blocks to extract features at several spatial scales.

[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

GoogLeNet is an image classification network based on the Inception module. It won the ImageNet classification challenge in 2014 and introduced a practical multi-branch structure for extracting features at different receptive fields.

- **Paper**: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- **Reference Implementation**: [torchvision/models/googlenet.py](https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py)

Feature highlights:

- **Inception module** — extracts multi-scale features using parallel
  convolution and pooling branches (1×1 / 3×3 / 5×5 convolutions and 3×3
  max pooling, concatenated per module).
- **Parameter efficiency** — reduces model parameters compared with
  wider dense CNN designs.
- **Deep architecture** — a 22-layer classification backbone with
  efficient branch aggregation.
- **Embedded deployment** — the RDK X5 deployment model uses packed NV12
  input and a quantized `.bin` artifact.

![Inception module: naive version and with dimension
reductions](./test_data/GoogLeNet_architecture.png)

*Figure (upstream paper Fig. 2): (a) the naive Inception module;
(b) the Inception module with dimension reductions — 1×1 convolutions
before the 3×3/5×5 branches and after pooling, which keeps compute
affordable at scale. The figure shows the upstream training
architecture; the deployed artifact is the INT8-quantized `googlenet`
variant at 224×224 NV12 (see [Support matrix](#support-matrix)).*

One BGR image produces ImageNet-1k Top-K class IDs, scores and optional
labels. The `GoogLeNetClassifier` class runs a `preprocess → infer →
postprocess` flow chained by `predict` (labels, drawing and file output
belong to the CLI layer; see
[runtime/python/README.md](runtime/python/README.md)).

<a id="directory"></a>
## Directory structure

```text
googlenet/
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
| x5 | googlenet | supported | not-supported |
| s100 | googlenet | not-supported | not-supported |
| s100p | googlenet | not-supported | not-supported |
| s600 | googlenet | not-supported | not-supported |

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
python3 -m venv .venv-googlenet
source .venv-googlenet/bin/activate
python3 -m pip install -r samples/vision/googlenet/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

From the repository root, prepare the X5 artifact with the model downloader, then run inference with the bundled image and the selected artifact reference.

```bash
# cwd: repository root
bash samples/vision/googlenet/model/download.sh x5 googlenet
python3 samples/vision/googlenet/runtime/python/main.py \
  --target x5 --variant googlenet \
  --test-img samples/vision/googlenet/test_data/indigo_bunting.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="expected-results"></a>
## Expected results

The default variant is `googlenet`. Inference prints Top-5 IDs, softmax scores and labels. Exact ties use ascending class-ID order. The bundled image is a functional input. Pass `--img-save-path` to save an image; otherwise results are printed to stdout.

The screenshot below shows a reference run from the X5 release: the demo
overlay draws the top-5 ranks onto the bundled `indigo_bunting.JPEG`,
with rank 1 being class 14 (indigo bunting).

![Reference inference result on X5: indigo bunting test image with the
top-5 overlay, rank 1 class 14 (indigo bunting)](./test_data/inference.png)

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
