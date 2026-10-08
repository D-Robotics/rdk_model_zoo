# VargConvNet image classification

<a id="overview"></a>

## Overview

VargConvNet is a lightweight convolutional classification model used for ImageNet-1k image classification on edge devices. The RDK X5 sample provides a prebuilt packed-NV12 `.bin` model and a Python runtime based on `hbm_runtime`.

The delivered model helper defines the runtime input and output contract; see the conversion guide before preparing replacement model files.

One BGR image produces ImageNet-1k Top-K class IDs, scores and optional
labels. The `VargConvNetClassifier` class runs a `preprocess → infer → postprocess` flow
chained by `predict` (labels, drawing and file output belong to the CLI
layer; see [runtime/python/README.md](runtime/python/README.md)).

<a id="directory"></a>
## Directory structure

```text
vargconvnet/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python and native inference implementations
├── test_data/  # Example inputs
├── tests/  # Automated tests
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── requirements-host.txt  # Source or data file
```

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Python | C++ |
| --- | --- | --- | --- |
| x5 | vargconvnet | supported | not-supported |
| s100 | vargconvnet | not-supported | not-supported |
| s100p | vargconvnet | not-supported | not-supported |
| s600 | vargconvnet | not-supported | not-supported |

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
python3 -m venv .venv-vargconvnet
source .venv-vargconvnet/bin/activate
python3 -m pip install -r samples/vision/vargconvnet/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

From the repository root, prepare the X5 artifact with the model downloader, then run inference with the bundled image and the selected artifact reference.

```bash
# cwd: repository root
bash samples/vision/vargconvnet/model/download.sh x5 vargconvnet
python3 samples/vision/vargconvnet/runtime/python/main.py \
  --target x5 --variant vargconvnet \
  --test-img samples/vision/vargconvnet/test_data/box_turtle.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="expected-results"></a>
## Expected results

The default variant is `vargconvnet`. Inference prints Top-5 IDs, softmax scores and labels. Exact ties use ascending class-ID order. The bundled image is a functional input. Pass `--img-save-path` to save an image; otherwise results are printed to stdout.

The screenshot below shows a reference run from the X5 release: the
demo overlay draws the top-5 ranks onto the bundled test image, with
rank 1 being class 37 (box turtle, box tortoise) with score 0.8582.

![Reference inference result on X5: box turtle test image with the top-5 overlay, rank 1 class 37 (box turtle, box tortoise) score 0.8582](./test_data/inference.png)

<a id="performance"></a>
## Performance data

Measure dataset accuracy and latency using the procedure under
[evaluation](evaluator/README.md#reference-results).

<a id="entry-points"></a>
## Entry points

[Model](model/README.md) · [Python](runtime/python/README.md) · [Conversion](conversion/README.md) · [Evaluation](evaluator/README.md)

<a id="license"></a>
## License

Python code follows Apache-2.0. Follow the original conversion notices and applicable upstream model and weight license terms. Review the upstream model and weight license terms before redistribution.
