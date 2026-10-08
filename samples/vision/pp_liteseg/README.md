English | [简体中文](README_cn.md)

# PP-LiteSeg-STDC1 semantic segmentation

<a id="overview"></a>
## Overview

PP-LiteSeg is a lightweight semantic segmentation network. The STDC1 variant in this sample labels road scenes with the 19 Cityscapes classes and runs on RDK X5.

References: [paper](https://arxiv.org/abs/2204.02681), [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg).

<a id="directory"></a>
## Directory structure

```text
pp_liteseg/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python and native inference implementations
├── test_data/  # Example inputs
├── tests/  # Automated tests
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Python | C++ |
| --- | --- | --- | --- |
| x5 | STDC1 / Cityscapes / 1024×512 | supported | not-supported |
| s100 / s100p / s600 | none published | not-supported | not-supported |

Board inference requires the X5 SDK and the matching published BIN.

<a id="prerequisites"></a>
## Prerequisites

RDK X5 OS 3.5.0+, Python 3.10+, board-provided hbm_runtime, NumPy, OpenCV and PyYAML. Host inspection does not load the SDK. Conversion uses the source OE 1.2.8 recipe in a separate container. Publisher model size and peak runtime memory are not recorded; reserve space for the BIN and outputs and measure memory on your board. One calibration tensor uses 6,291,456 bytes.

<a id="quickstart"></a>
## Quick start

```bash
# cwd: repository root; prepare explicitly, then run on RDK X5
bash samples/vision/pp_liteseg/model/download.sh --target x5
python3 samples/vision/pp_liteseg/runtime/python/main.py
# Host-only inspection, no SDK/model/download required:
python3 samples/vision/pp_liteseg/runtime/python/main.py --dry-run --target x5
```

Install general Python dependencies with `python3 -m pip install numpy opencv-python PyYAML`. Inference never downloads implicitly; run.sh is an argument-forwarding helper.

<a id="expected-results"></a>
## Expected results

Success returns 0 and writes `outputs/pp_liteseg/result.jpg` (3078×548, Original / Overlay / Segmentation), `labels.npy` (512×1024 int32 IDs 0..18) and `result.json` in the same directory. JSON/stdout report actual class names and runtime metadata. No class list or accuracy is promised for the supplied street image without real inference. Errors return 2. Mask coordinates refer to the stretched model input, not original image dimensions.

The compiled model returns a decoded int32 class map. `postprocess` validates class IDs and removes batch/channel dimensions; CPU argmax is unnecessary. Model metadata is checked when loading.

<a id="entry-points"></a>
## Entry points

[Model preparation](model/README.md) · [Python CLI and API](runtime/python/README.md) · [Conversion](conversion/README.md) · [Validation](evaluator/README.md).

<a id="license"></a>
## License

Repository code follows the top-level [LICENSE](../../../LICENSE); retained source notices remain applicable. PaddleSeg and externally obtained pretrained weights/data have their own terms. The manifest does not establish the pretrained weight license; check the actual checkpoint source before redistribution.
