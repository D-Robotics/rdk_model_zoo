English | [简体中文](README_cn.md)

# MobileSAM

<a id="overview"></a>
## Overview

MobileSAM replaces SAM’s image encoder with TinyViT to reduce the cost of prompt-based segmentation. This sample uses one box prompt to select a mask from the decoder’s candidates.

References: <https://arxiv.org/abs/2306.14289>, <https://github.com/ChaoningZhang/MobileSAM>.

<a id="directory"></a>
## Directory structure

```text
mobile_sam/
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

From the repository root, prepare the pair and run on the matching board. The default box is `[185,120,380,445]` in resized `512x512` coordinates.

```bash
# cwd: repository root; prerequisite: network access only for this preparation step
python3 samples/vision/mobile_sam/model/download.py --target s100
# expect: model/nash-e/mobile_sam_image_encoder_norm_512x512_nashe.hbm and decoder_512_nashe.hbm

# cwd: repository root; prerequisite: prepared pair and matching board runtime
python3 samples/vision/mobile_sam/runtime/python/main.py --target s100 --box 185,120,380,445
# expect: JSON on stdout and mobile_sam_full_mask_result.jpg plus mobile_sam_binary_mask_result.png under test_data/
```

<a id="expected-results"></a>
## Expected Results

The default image is `test_data/dogs.jpg`. A successful run writes a `512x512` overlay and binary mask. IoU and mask index are model outputs; `mobile_sam_binary_mask.png` is the preserved source reference.

The input is stretched to 512×512; the `[x1,y1,x2,y2]` box and output mask use that coordinate system. RGB normalization uses mean `[123.675,116.28,103.53]` and std `[58.395,57.12,57.375]`. The encoder produces `(1,256,32,32)` embeddings. The decoder returns three mask candidates with IoU scores, and the selected mask uses logits `>0`.

<a id="entry-points"></a>
## Entry Points

- [Model](model/README.md): target pair mapping, preparation and checksums.
- [Python runtime](runtime/python/README.md): CLI and `MobileSAMPipeline` API.
- [Conversion](conversion/README.md): export, calibration and compile material.
- [Evaluation](evaluator/README.md): evaluation procedure and reference records.
- C++: not provided.

<a id="license"></a>
## License

Runtime code follows the repository Apache-2.0 license. MobileSAM checkpoints and ONNX/model assets retain upstream terms; verify them before redistribution.
