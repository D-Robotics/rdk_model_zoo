English | [简体中文](README_cn.md)

# UNetMobileNet semantic segmentation

<a id="overview"></a>
## Overview

UNetMobileNet combines a U-Net encoder/decoder with a lightweight MobileNet backbone for Cityscapes 19-class segmentation. Preserved algorithm references: [U-Net paper](https://arxiv.org/abs/1505.04597), [MobileNet paper](https://arxiv.org/abs/1704.04861), [Cityscapes](https://www.cityscapes-dataset.com/). The source does not identify an exact training repository/checkpoint release.

This S-family sample differs from X5 UNet: two NV12 input planes at 2048×1024, INTER_AREA stretch, 19 classes, original-resolution output. Both Python and C++ separate preprocessing, raw forward and mask decoding; rendering lives outside predict.

<a id="support-matrix"></a>
## Support and validation

| Target | Variant | Python | C++ |
| --- | --- | --- | --- |
| s100 | unet_mobilenet_1024x2048_nv12, S100 HBM | supported-not-run | supported-not-run |
| s600 | same model family, separate S600 HBM | supported-not-run | supported-not-run |
| x5 / s100p | no published asset | not-supported | not-supported |

Host fixtures verify stages, selection and CLI; pure C++ tests verify decoding and resource cleanup with fake SDK interfaces. Real SDK compilation/inference, board tests, dataset accuracy and performance remain not-run. [Source audit](../../../docs/releases/unified-migration/evidence/2026-09-26-b8-unetmobilenet-audit.json).

<a id="prerequisites"></a>
## Prerequisites

S100 or S600 with its matching board image and hbm_runtime (Python) or DNN/UCP headers/libraries (C++). The source does not pin a minimum S OS/SDK version; none is invented here. Python 3.10+, NumPy, OpenCV-Python and PyYAML; C++17, CMake 3.16+ and OpenCV development libraries for native compilation. Runtimes never install packages or fetch weights. Model size and peak memory are unmeasured; full-resolution 19-channel int32 scores alone can occupy about 152 MiB, and float64 decoding needs additional memory.

<a id="quickstart"></a>
## Quick start

```bash
# cwd: repository root; run on the selected S100 board
bash samples/vision/unetmobilenet/model/download.sh --target s100
python3 samples/vision/unetmobilenet/runtime/python/main.py --target s100
# For S600, use --target s600 for BOTH preparation and inference.
# Host-only selection inspection, no SDK/model/download:
python3 samples/vision/unetmobilenet/runtime/python/main.py --dry-run --target s600
```

From the repository root, install general dependencies with `python3 -m pip install numpy opencv-python PyYAML`. On a recognized supported board, the zero-argument main.py command auto-selects its exact target. Unknown identities fail; S100P never falls back to S100.

<a id="expected-results"></a>
## Expected results

Python success returns 0 and writes result.jpg, unetmobilenet_mask.npy (original-size int32 IDs 0..18) and unetmobilenet_report.json in cwd. alpha_f=0.75 weights the original image, so 1 shows the original and 0 the mask colors. Actual classes depend on real inference; no fixed result is promised. The figure below is the retained source illustration, not a new board result.

![Historical source result](test_data/result.jpg)

<a id="directory"></a>
## Directory responsibilities

- model/: explicit target-scoped download into s100/ or s600/.
- runtime/python/: task stages, binding, shared runner adapter, CLI and visualization.
- runtime/cpp/: stage implementation, SDK resource owner, pure tensor decoder, launcher/build.
- conversion/: missing recipe prerequisites; no export/compiler implementation in source.
- evaluator/: single-image checks and dataset/performance boundaries; no dataset loop in source.
- test_data/: segmentation.png input and preserved historical result.jpg.
- tests/: source parity, host fixtures, CLI, native numerical/resource tests.

<a id="entry-points"></a>
## Entry points

[Model](model/README.md) · [Python](runtime/python/README.md) · [C++](runtime/cpp/README.md) · [Conversion](conversion/README.md) · [Validation](evaluator/README.md). [Original S documentation](../../../platforms/s/samples/vision/unetmobilenet/README.md) remains archived with its old API and automatic setup behavior.

<a id="license"></a>
## License

Code follows the repository [LICENSE](../../../LICENSE); source copyright notices are retained. The published manifest does not establish training-weight or Cityscapes redistribution rights; check those upstream terms separately.
