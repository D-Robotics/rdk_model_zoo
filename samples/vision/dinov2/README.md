English | [简体中文](./README_cn.md)

# DINOv2 ViT-S/14 vision features

<a id="overview"></a>
## Overview

DINOv2 is a self-supervised ViT encoder that produces a global image feature and dense patch features. This sample deploys the ViT-S/14 backbone as an int16 PTQ HBM model for RDK S100, S100P, and S600. The upstream implementation and Apache-2.0 model artifacts are [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2), pinned for conversion to revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.

The graph has a patch-14 stem, 12 pre-LN transformer blocks, explicit BPU-friendly attention, and a final normalized feature interface. The sample exposes `cls_feat` `(1,384)` for global embedding and `patch_feat` `(1,256,384)` for per-patch features. Runtime post-processing dequantizes integer outputs using the bound metadata into owned float32 arrays; it does not apply softmax or L2 normalization.

<a id="support-matrix"></a>
## Support Matrix

The one published variant has independent HBM artifacts for Nash-E, Nash-M, and Nash-P. `supported-not-run` means the local contract and fixture coverage exist but no board was used in this migration. Python is supported-not-run; no C++ runtime is provided.

| Variant | x5 | s100 | s100p | s600 | Python | C++ |
| --- | --- | --- | --- | --- | --- | --- |
| `vits14-224-int16` | not-supported | supported-not-run | supported-not-run | supported-not-run | supported-not-run | not-supported |

Board verification evidence: not-run. Conversion scripts are source-backed and documented, but were not executed in this migration.

<a id="prerequisites"></a>
## Prerequisites

- Board execution: RDK S100 (Nash-E), S100P (Nash-M), or S600 (Nash-P), with a board image that provides `hbm_runtime`. Board image and firmware versions were not verified.
- Host contract checks: Python 3.14.7 with `numpy`, `opencv-python`, and `PyYAML` from `requirements-host.txt`.
- Conversion: x86 Linux OE 3.7.0 image `ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0`; Torch 2.6 is supplied by that image, with `onnx==1.19.0` and `onnxruntime==1.23.2` additions. Conversion was not run here.
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

The CLI prints a JSON summary with `output`, `shape`, `dtype`, `mean`, `std`, `min`, `max`, and `l2_norm`. When the default second image exists it also prints `second_image` and `cosine_similarity`; missing second images are reported as `skipped_missing`. `cls_feat` has shape `(1,384)` and `patch_feat` `(1,256,384)`, both returned as float32 after metadata-bound dequantization. Exact board values are not claimed until a target board is run.

<a id="directory"></a>
## Directory Layout

```text
dinov2/
├── conversion/             # pinned ONNX export, calibration, and hb_compile recipe
│   └── onnx_export/        # source-model export and graph rewrites
├── evaluator/              # historical performance and cosine records
├── model/                  # manifest-backed target-specific HBM preparation
├── runtime/python/         # binding, runner, feature task, tensor I/O, and CLI
├── test_data/              # dog.jpg and bus.jpg inputs
└── README.md               # this guide
```

<a id="entry-points"></a>
## Entry Points

- Model preparation: [`model/README.md`](model/README.md) — three target-specific HBM artifacts and exact URLs.
- Python runtime: [`runtime/python/README.md`](runtime/python/README.md) — preprocessing, dual-output task API, and CLI.
- C++ runtime: not provided; C++ is `not-supported`.
- Conversion: [`conversion/README.md`](conversion/README.md) — pinned source, ONNX export, calibration, and compile commands.
- Evaluation: [`evaluator/README.md`](evaluator/README.md) — historical board tables and cosine reproduction conditions.

<a id="license"></a>
## License

The DINOv2 source model and checkpoint are Apache-2.0 artifacts published by Meta AI through [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2). Sample code follows the repository [LICENSE](../../../LICENSE), Apache-2.0. Source contributor attribution is the D-Robotics model zoo team.
