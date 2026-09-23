English | [简体中文](README_cn.md)

# MobileSAM

<a id="overview"></a>
## Overview

MobileSAM performs box-prompted image segmentation with a TinyViT image encoder and a mask decoder. The encoder produces a `1x256x32x32` embedding from a normalized `512x512` RGB image; the decoder consumes that embedding and one `[x1,y1,x2,y2]` box, returns three mask candidates and IoU scores, and the runtime selects and upsamples the best candidate.

- Paper: <https://arxiv.org/abs/2306.14289>
- Official repository: <https://github.com/ChaoningZhang/MobileSAM>
- Source baseline: `platforms/s/samples/vision/mobile_sam` and `platforms/x5/samples/vision/mobile_sam`

The input is stretched directly to 512×512. The box and result mask use that resized coordinate system; no inverse transform to the original image is performed. RGB values use mean `[123.675,116.28,103.53]` and std `[58.395,57.12,57.375]`; the selected mask uses logits `>0`.

<a id="support-matrix"></a>
## Support Matrix

| Target | Variant | Python | C++ |
|---|---|---|---|
| x5 | default `.bin` pair | supported-not-run | not-supported |
| s100 | nash-e `.hbm` pair | supported-not-run | not-supported |
| s100p | nash-m `.hbm` pair | supported-not-run | not-supported |
| s600 | nash-p `.hbm` pair | supported-not-run | not-supported |

Host fixtures use injected runners. Board execution and `hbm_runtime` compatibility remain not-run.

<a id="prerequisites"></a>
## Prerequisites

Host checks use the repository `.venv`, Python, NumPy, OpenCV and PyYAML. The recorded host fixture is Python 3.14.7, NumPy 2.5.3, OpenCV 4.14.0 and PyYAML 6.0.3; NumPy/OpenCV versions are not pinned by this sample. Runtime syntax requires Python 3.10 or newer. A matching RDK image and `hbm_runtime` are required on board; their system version is unknown and not-run. Prepare both model assets explicitly; inference never downloads. Conversion material is in [conversion](conversion/README.md).

The board's SDK must already be installed by its matching system image; do not install `hbm_runtime` from an unrelated host environment. Check the required Python imports from the repository root:

```bash
# cwd: repository root on the selected board
python3 -c "import numpy, cv2, yaml, hbm_runtime; print('runtime dependencies available')"
```

If only the ordinary Python dependencies are missing, install them in the Python environment used by that board's SDK (`python3 -m pip install numpy opencv-python PyYAML`). The source does not pin their board versions; preserve the image's SDK compatibility constraints. The command above checks import availability only. Disk/RAM requirements were not measured; both encoder and decoder must fit in the target runtime.

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

<a id="directory"></a>
## Directory

```text
mobile_sam/
├── model/                 # manifest-backed encoder/decoder preparation
├── runtime/python/        # binding, runner, pipeline, CLI, visualization
├── test_data/             # dogs image and preserved source mask
├── conversion/            # source conversion material
├── evaluator/             # evaluation procedure and limits
├── README.md              # English overview
└── README_cn.md           # Chinese overview
```

<a id="entry-points"></a>
## Entry Points

- [Model](model/README.md): target pair mapping, preparation and checksums.
- [Python runtime](runtime/python/README.md): CLI and `MobileSAMPipeline` API.
- [Conversion](conversion/README.md): export, calibration and compile material.
- [Evaluation](evaluator/README.md): reference procedure and not-run boundary.
- C++: not provided.

<a id="license"></a>
## License

Runtime code follows the repository Apache-2.0 license. MobileSAM checkpoints and ONNX/model assets retain upstream terms; verify them before redistribution.
