English | [简体中文](README_cn.md)

# PP-LiteSeg-STDC1 semantic segmentation

<a id="overview"></a>
## Overview

PP-LiteSeg-STDC1 predicts the 19 Cityscapes road-scene classes for every pixel. This sample consolidates the X5 Python inference, conversion recipe and single-image validation entry. Algorithm references retained from the source: [paper](https://arxiv.org/abs/2204.02681), [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg).

The source **runtime** consumes an already-decoded int32 class map. Older root/conversion READMEs described logits plus CPU argmax; that description does not match the delivered runtime. Here post_process only validates and removes batch/channel dimensions. Actual compiled artifact metadata still requires board validation.

<a id="support-matrix"></a>
## Support and verification

| Target | Variant | Python | C++ |
| --- | --- | --- | --- |
| x5 | STDC1 / Cityscapes / 1024×512 | supported-not-run | not-supported |
| s100 / s100p / s600 | none published | not-supported | not-supported |

Host fixtures verify source preprocessing, class-map decoding, rendering, selection and CLI behavior. They do not certify the SDK or the published BIN. Board tests, dataset metrics and conversion execution: not-run. Source audit: [evidence](../../../docs/releases/unified-migration/evidence/2026-09-26-b8-ppliteseg-audit.json).

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

Install general Python dependencies with `python3 -m pip install numpy opencv-python PyYAML`. Inference never downloads implicitly; run.sh is an argument-forwarding wrapper.

<a id="expected-results"></a>
## Expected results

Success returns 0 and writes `outputs/pp_liteseg/result.jpg` (3078×548, Original / Overlay / Segmentation), `labels.npy` (512×1024 int32 IDs 0..18) and `result.json` in the same directory. JSON/stdout report actual class names and runtime metadata. No class list or accuracy is promised for the supplied street image without real inference. Errors return 2. Mask coordinates refer to the stretched model input, not original image dimensions.

<a id="directory"></a>
## Directory responsibilities

- `model/`: explicit published-asset preparation; weights are not bundled.
- `runtime/python/`: four-stage task, binding, SDK runner, CLI and visualization.
- `conversion/`: PaddleSeg export, raw calibration preparation and OE YAML/build.
- `evaluator/`: single-image compatibility CLI and validation boundaries.
- `test_data/`: source `street.png` and `test.jpg`; the old `street.jpg` path was absent.
- `tests/`: host fixtures and source-parity checks.

<a id="entry-points"></a>
## Entry points

[Model preparation](model/README.md) · [Python CLI and API](runtime/python/README.md) · [Conversion](conversion/README.md) · [Validation](evaluator/README.md). The original X5 snapshot remains under [platforms/x5](../../../platforms/x5/samples/vision/pp_liteseg/README.md).

<a id="license"></a>
## License

Repository code follows the top-level [LICENSE](../../../LICENSE); retained source notices remain applicable. PaddleSeg and externally obtained pretrained weights/data have their own terms. The manifest does not establish the pretrained weight license; check the actual checkpoint source before redistribution.
