English | [简体中文](README_cn.md)

# EfficientSAM-Tiny

<a id="overview"></a>
## Overview

EfficientSAM-Tiny segments one image using two fixed positive point prompts baked into its decoder with a ViT-Tiny image encoder and a fixed-prompt mask decoder. The encoder produces a `1x256x32x32` embedding from a normalized `512x512` RGB image; the decoder produces three low-resolution mask candidates and IoU scores. The selected mask is resized to `512x512` and returned as a binary mask.

- Paper: <https://arxiv.org/abs/2312.00863>
- Project: <https://yformer.github.io/efficient-sam/>

The published decoder fixes positive points `(248,210)` and `(302,315)` in the resized 512-square image. It accepts no runtime point or box argument. The encoder applies RGB `/255`; the selected mask uses logits `>=0`. Input is stretched to 512×512 and the result stays in that coordinate system, without mapping back to the original image.

<a id="directory"></a>
## Directory structure

```text
efficient_sam/
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
## Support Matrix

| Target | Variant | Python | C++ |
|---|---|---|---|
| x5 | default `.bin` pair | supported | not-supported |
| s100 | nash-e `.hbm` pair | supported | not-supported |
| s100p | nash-m `.hbm` pair | supported | not-supported |
| s600 | nash-p `.hbm` pair | supported | not-supported |

Board inference requires the selected target board and its `hbm_runtime`.

<a id="prerequisites"></a>
## Prerequisites

Use Python 3.10+, NumPy, OpenCV and PyYAML in the target RDK image’s `hbm_runtime` environment. Prepare both encoder and decoder model files before inference. See the conversion guide for the target OE toolchain.

The board's SDK must already be installed by its matching system image; do not install `hbm_runtime` from an unrelated host environment. Check the required Python imports from the repository root:

```bash
# cwd: repository root on the selected board
python3 -c "import numpy, cv2, yaml, hbm_runtime; print('runtime dependencies available')"
```

Install Python dependencies in the board SDK environment (`python3 -m pip install numpy opencv-python PyYAML`). Reserve memory for both encoder and decoder to remain loaded simultaneously.

<a id="quickstart"></a>
## Quick Start

From the repository root, prepare a pair and run on a matching board. Replace `s100` with `x5`, `s100p`, or `s600` and provide the corresponding target asset IDs when using custom paths.

```bash
# cwd: repository root; prerequisite: network access only for this preparation step
python3 samples/vision/efficient_sam/model/download.py --target s100
# expect: model/nash-e/efficient_sam_vitt_encoder_512x512_nashe.hbm and decoder_512_nashe.hbm

# cwd: repository root; prerequisite: prepared pair and matching board runtime
python3 samples/vision/efficient_sam/runtime/python/main.py --target s100
# expect: JSON on stdout, efficient_sam_full_mask_result.jpg and efficient_sam_binary_mask_result.png under test_data/
```

<a id="expected-results"></a>
## Expected Results

The default input is `test_data/dogs.jpg`. A successful run writes an overlay image and a binary `512x512` mask. The numeric IoU and selected mask index depend on the actual model execution; the committed `efficient_sam_binary_mask.png` is a source reference, not a new board result.

<a id="entry-points"></a>
## Entry Points

- [Model](model/README.md): target pair mapping, preparation and checksums.
- [Python runtime](runtime/python/README.md): CLI and `EfficientSAMPipeline` API.
- [Conversion](conversion/README.md): export, calibration and compile material.
- [Evaluation](evaluator/README.md): evaluation procedure and reference records.
- C++: not provided.

<a id="license"></a>
## License

Runtime code follows the repository Apache-2.0 license. EfficientSAM checkpoints and ONNX/model assets retain the upstream project terms; verify those terms before redistribution.
