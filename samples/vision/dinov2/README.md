English | [简体中文](./README_cn.md)

# DINOv2 ViT-S/14 vision features

<a id="overview"></a>
## Overview

DINOv2 is a self-supervised vision transformer for image representations. The ViT-S/14 backbone produces a global embedding and dense patch features for downstream visual tasks. This sample deploys its int16 PTQ model on RDK S100, S100P and S600.

References: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).

<a id="directory"></a>
## Directory structure

```text
dinov2/
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
## Support Matrix

The one published variant has independent HBM artifacts for Nash-E, Nash-M, and Nash-P. Python is supported; no C++ runtime is provided.

| Variant | x5 | s100 | s100p | s600 | Python | C++ |
| --- | --- | --- | --- | --- | --- | --- |
| `vits14-224-int16` | not-supported | supported | supported | supported | supported | not-supported |

Board execution requires the matching S-series board with `hbm_runtime`; the conversion scripts are documented in [conversion](conversion/README.md).

<a id="prerequisites"></a>
## Prerequisites

- Board execution: RDK S100 (Nash-E), S100P (Nash-M), or S600 (Nash-P), with a board image that provides `hbm_runtime`. Use the Python environment supplied with that board image.
- Python dependencies: Python 3.10+ with `numpy`, `opencv-python`, and `PyYAML` from `requirements-host.txt`.
- Conversion: x86 Linux OE 3.7.0 image `ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0`; Torch 2.6 is supplied by that image, with `onnx==1.19.0` and `onnxruntime==1.23.2` additions.
- Prepare one target-specific HBM before board inference; runtime commands do not download implicitly.

<a id="quickstart"></a>
## Quick Start

From the repository root, explicitly prepare the S100 artifact and then run the board CLI. Use `s100p` or `s600` to select the corresponding independent artifact.

```bash
# cwd: repository root; source: exact URL in docs/release/s/models.yaml
python3 samples/vision/dinov2/model/download.py --target s100
# expect: samples/vision/dinov2/model/nash-e/dinov2_vits14_224_int16_nashe.hbm

# cwd: repository root; input: test_data/dog.jpg and test_data/bus.jpg
python3 samples/vision/dinov2/runtime/python/main.py --target s100 --output cls_feat
# expect: JSON summary for cls_feat and cosine_similarity for the second image; exit code 0
```

The convenience `runtime/python/run.sh` accepts the positional output (`cls_feat` or `patch_feat`) followed by named options. It never downloads a model. The source-compatible `model/download_model.sh` delegates to the explicit target script and requires `s100`, `s100p`, or `s600`.

<a id="expected-results"></a>
## Expected Results

The CLI prints a JSON summary with `output`, `shape`, `dtype`, `mean`, `std`, `min`, `max`, and `l2_norm`. When the default second image exists it also prints `second_image` and `cosine_similarity`; missing second images are reported as `skipped_missing`. `cls_feat` has shape `(1,384)` and `patch_feat` `(1,256,384)`, both returned as float32 after metadata-bound dequantization. Board values are obtained by running the target board.

Outputs are `cls_feat` `(1,384)` and `patch_feat` `(1,256,384)`, returned as owned float32 arrays after metadata-based dequantization. The runtime applies neither softmax nor L2 normalization. The backbone uses a patch-14 stem and 12 pre-LN transformer blocks.

<a id="entry-points"></a>
## Entry Points

- Model preparation: [`model/README.md`](model/README.md) — three target-specific HBM artifacts and exact URLs.
- Python runtime: [`runtime/python/README.md`](runtime/python/README.md) — preprocessing, dual-output task API, and CLI.
- C++ runtime: not provided; C++ is `not-supported`.
- Conversion: [`conversion/README.md`](conversion/README.md) — pinned source, ONNX export, calibration, and compile commands.
- Evaluation: [`evaluator/README.md`](evaluator/README.md) — source-recorded board tables and cosine reproduction conditions.

<a id="license"></a>
## License

The DINOv2 source model and checkpoint are Apache-2.0 artifacts published by Meta AI through [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2). Sample code follows the repository [LICENSE](../../../LICENSE), Apache-2.0. Source contributor attribution is the D-Robotics model zoo team.
