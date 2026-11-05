English | [简体中文](README_cn.md)

# RepGhost image classification

RepGhost uses structural reparameterization to replace explicit feature concatenation in a lightweight CNN.

[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

RepGhost is a lightweight CNN family designed to improve hardware efficiency by replacing explicit feature reuse in feature space with re-parameterized reuse in weight space. It avoids costly `Concat` operations while keeping strong classification performance.

- **Paper**: [RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization](https://arxiv.org/abs/2211.06088)
- **Reference Implementation**: [ChengpengChen/RepGhost](https://github.com/ChengpengChen/RepGhost)

Feature highlights:

- **Structural re-parameterization** — converts training-time complex
  branches into efficient inference-time structures.
- **Implicit feature reuse** — moves GhostNet-style feature reuse from
  feature space (`Concat`) to weight space, avoiding costly memory
  copies.
- **Hardware efficiency** — reduces memory-copy overhead and improves
  deployment efficiency on edge devices.
- **Variant scaling** — published variants from `100` to `200`.

![RepGhost bottleneck compared with the Ghost bottleneck: training-time
add branches fused for inference](./test_data/RepGhost_architecture.png)

*Figure (upstream paper Fig. 4): (a) Ghost bottleneck with its explicit
`Concat` feature reuse; (b) RG-bneck at training — reuse moves to weight
space via `add` branches; (c) RG-bneck at inference — the branches are
fused away. It depicts the upstream
architecture, while the deployed artifacts are the INT8-quantized
100–200 variants at 224×224 NV12 (see [Support
matrix](#support-matrix)).*

One BGR image produces ImageNet-1k Top-K class IDs, scores and optional
labels. The `RepGhostClassifier` class runs a `preprocess → infer → postprocess` flow
chained by `predict` (labels, drawing and file output belong to the CLI
layer; see [runtime/python/README.md](runtime/python/README.md)).

<a id="directory"></a>
## Directory structure

```text
repghost/
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

The table lists the models published for this sample; `—` marks an
unpublished combination.

| Model | Input | X5 | S100 | S100P | S600 |
| --- | --- | --- | --- | --- | --- |
| RepGhost-100 | 224x224 | ✓ | — | — | — |
| RepGhost-111 | 224x224 | ✓ | — | — | — |
| RepGhost-130 | 224x224 | ✓ | — | — | — |
| RepGhost-150 | 224x224 | ✓ | — | — | — |
| RepGhost-200 | 224x224 | ✓ | — | — | — |

Runtime coverage: the Python CLI serves every published combination with
`--target auto|x5` and `--variant 100|111|130|150|200` (default `100`);
each lowercase variant ID resolves to the exact case-sensitive published
filename `RepGhost_<width>_224x224_nv12.bin`. `--target s100`, `s100p` and
`s600` resolve no asset. The sample ships no C++ runtime.

<a id="prerequisites"></a>
## Prerequisites

Use a full repository checkout. On X5 use the matching board image and its `hbm_runtime`; install host dependencies in a virtual environment as below. Host-side comparison tools use SciPy; board inference uses the dependencies in the matching runtime image.

Use a full repository checkout. On X5 use the matching board image and
its `hbm_runtime`; install the host dependencies in a virtual environment
as below. Host-side comparison tools use SciPy; board inference uses the dependencies in the matching runtime image. Native inference needs no OE toolchain; conversion
prerequisites are documented under [conversion](conversion/README.md).
```bash
# cwd: repository root
python3 -m venv .venv-repghost
source .venv-repghost/bin/activate
python3 -m pip install -r samples/vision/repghost/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

From the repository root, prepare the X5 artifact with the model downloader, then run inference with the bundled image and the selected artifact reference.

```bash
# cwd: repository root
bash samples/vision/repghost/model/download.sh x5 100
python3 samples/vision/repghost/runtime/python/main.py \
  --target x5 --variant 100 \
  --test-img samples/vision/repghost/test_data/ibex.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="expected-results"></a>
## Expected results

The default variant is `100`; `111`, `130`, `150`, `200` must be selected
explicitly. Softmax scores produce a stable Top-K; exact ties use stable
ascending class-ID order. For dataset accuracy, use the validation-set procedure in the evaluator guide. No image is saved unless `--img-save-path` is
supplied.

The screenshot below shows a reference run from the X5 release: the
demo overlay draws the top-5 ranks onto the bundled test image, with
rank 1 being class 350 (ibex, Capra ibex).

![Reference inference result on X5: ibex test image with the top-5 overlay, rank 1 class 350 (ibex, Capra ibex)](./test_data/inference.png)

<a id="performance"></a>
## Performance data

X5 records from the x5-v1.1.3 release; full conditions and all rows are in
[evaluator](evaluator/README.md#reference-results).

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread latency (ms) | Multi-thread latency (ms) | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RepGhost_200 | 224x224 | 9.79 | 76.43 | 75.25 | 2.89 | 8.76 | 451.42 |
| RepGhost_150 | 224x224 | 6.57 | 74.75 | 73.50 | 2.20 | 6.30 | 626.60 |
| RepGhost_130 | 224x224 | 5.48 | 75.00 | 73.57 | 1.87 | 5.30 | 743.56 |
| RepGhost_111 | 224x224 | 4.54 | 72.75 | 71.25 | 1.71 | 4.47 | 881.19 |
| RepGhost_100 | 224x224 | 4.07 | 72.50 | 72.25 | 1.55 | 4.08 | 964.69 |

<a id="entry-points"></a>
## Entry points

[Model](model/README.md) · [Python](runtime/python/README.md) · [Conversion](conversion/README.md) · [Evaluation](evaluator/README.md)

<a id="license"></a>
## License

Source Python headers retain Apache-2.0 provenance. Conversion YAMLs retain their original proprietary notices verbatim; before redistribution, follow the original conversion YAML notices and applicable upstream weight license terms.

<!-- converted-batch-20261102 -->
## Converted-only additions (2026-11-02)

These independently converted timm weights have passed strict checkpoint loading,
reparameterization and two-input PyTorch/ONNX host comparison, checker and compiler.
Board smoke, accuracy, Runtime 1/2 and C++ 1/2 are **not-run**; historical metrics
and screenshots do not describe these artifacts. Existing defaults are retained.
Pillow is needed for the explicit bicubic shorter-edge 256 / rounded center crop
224 profile. Normalization is fused in Mapper from the actual checkpoint config;
output is float32 1,000-class logits followed by native softmax.

| Variant | Artifact | Checkpoint revision | Board |
|---|---|---|---|
| `timm-100-in1k` | `repghostnet_100_in1k_bayese_224x224_nv12.bin` | [`3c79eee5f138`](https://huggingface.co/timm/repghostnet_100.in1k/blob/3c79eee5f138318c2cb58957105b91bb50259b29/README.md) | not-run |
| `timm-111-in1k` | `repghostnet_111_in1k_bayese_224x224_nv12.bin` | [`03ffb284f8ea`](https://huggingface.co/timm/repghostnet_111.in1k/blob/03ffb284f8ea7ae1c41a1c2617d4dab98d7cd9f4/README.md) | not-run |
| `timm-130-in1k` | `repghostnet_130_in1k_bayese_224x224_nv12.bin` | [`efb0a7881fad`](https://huggingface.co/timm/repghostnet_130.in1k/blob/efb0a7881fada70cec140ce2810f9c0a7d416704/README.md) | not-run |
| `timm-150-in1k` | `repghostnet_150_in1k_bayese_224x224_nv12.bin` | [`577d935410a6`](https://huggingface.co/timm/repghostnet_150.in1k/blob/577d935410a68c6efc6a0fee9f11f88f942ac690/README.md) | not-run |
| `timm-200-in1k` | `repghostnet_200_in1k_bayese_224x224_nv12.bin` | [`0c520f5e8de7`](https://huggingface.co/timm/repghostnet_200.in1k/blob/0c520f5e8de77f9bfedc23f2104c344b1d685530/README.md) | not-run |

```bash
python3 samples/vision/repghost/model/download.py --target x5 --variant timm-100-in1k
python3 samples/vision/repghost/runtime/python/main.py --target x5 --variant timm-100-in1k --dry-run
```

Exact model URLs/SHA-256/size are in `docs/release/x5/models.yaml`; source and host
receipts are in `conversion/converted.json`. Each weight retains its own model
license in its public release manifest. No board result is implied by download.
<!-- /converted-batch-20261102 -->

<!-- converted-all-20261103 -->

## Converted S profiles (2026-11-03)

Independent converted profiles and explicit commands are listed in [model preparation](model/README.md). Board accuracy and performance are not measured yet.
