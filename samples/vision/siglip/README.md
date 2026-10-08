English | [简体中文](./README_cn.md)

# SigLIP vision features

<a id="overview"></a>
## Overview

SigLIP is a vision encoder that turns one image into a global embedding or a sequence of patch features. This sample exposes only the vision side of SigLIP: it does not include a text encoder or text-token pipeline. The upstream paper is [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343); the model family is published by [Google Research](https://github.com/google-research/big_vision). The sample lives at `samples/vision/siglip`.

The eight published variants are packed HBM artifacts. Every artifact contains two fixed submodels, `pooler_output` and `last_hidden_state`, with the same image input and selected-output execution.

<a id="directory"></a>
## Directory structure

```text
siglip/
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

All eight variants and both submodels are `supported` on S100 and S100P; the HBM filenames under `s100/` are the shared publication assets and do not remove S100P support. Python is `supported`; C++ has no implementation and is `not-supported`. X5 and S600 are `not-supported`.

| Variant | x5 | s100 | s100p | s600 | Python | C++ |
| --- | --- | --- | --- | --- | --- | --- |
| `base-patch16-224` | not-supported | supported | supported | not-supported | supported | not-supported |
| `base-patch16-384` | not-supported | supported | supported | not-supported | supported | not-supported |
| `base-patch16-512` | not-supported | supported | supported | not-supported | supported | not-supported |
| `large-patch16-256` | not-supported | supported | supported | not-supported | supported | not-supported |
| `large-patch16-384` | not-supported | supported | supported | not-supported | supported | not-supported |
| `so400m-patch14-224` | not-supported | supported | supported | not-supported | supported | not-supported |
| `so400m-patch14-384` | not-supported | supported | supported | not-supported | supported | not-supported |
| `so400m-patch16-256-i18n` | not-supported | supported | supported | not-supported | supported | not-supported |

Board execution requires a board image with `hbm_runtime`; host tests cover contracts and injected fixtures only.

<a id="prerequisites"></a>
## Prerequisites

- Board: RDK S100 or S100P with a board image providing `hbm_runtime`; no image or BPU firmware version is pinned.
- Host preparation: Python 3.14.7 with `numpy`, `opencv-python`, and `PyYAML` from `requirements-host.txt`.
- The HBM must be prepared before inference. No runtime command downloads a model implicitly.
- The source reports no memory or disk requirement beyond the selected HBM file.

<a id="quickstart"></a>
## Quick Start

The following is the complete explicit path. It requires network access for the first command and a local S100/S100P board for the second.

```bash
# cwd: repository root; source: the release manifest URL in model/README.md
python3 samples/vision/siglip/model/download.py --target s100 --variant base-patch16-224
# expect: samples/vision/siglip/model/s100/bpu-siglip-base-patch16-224.hbm

# cwd: repository root; input: samples/vision/siglip/test_data/dog.jpg
python3 samples/vision/siglip/runtime/python/main.py --target s100 --variant base-patch16-224 --submodel pooler_output
# expect: JSON summary with submodel, shape, dtype, mean, std, min, max, and l2_norm; exit code 0
```

The convenience `runtime/python/run.sh` accepts the source positional submodel (`pooler_output` or `last_hidden_state`) and then named options. It does not download models.

<a id="expected-results"></a>
## Expected Results

The CLI prints one JSON statistics object for the selected raw feature tensor. Its `shape` is bound from HBM metadata: `pooler_output` is `(1,D)` or `(1,1,D)`, and `last_hidden_state` is `(1,N,D)`. `D` and `N` are listed in the runtime README. The native output dtype is preserved; the runtime does not dequantize, apply softmax, normalize, squeeze, or otherwise alter the feature values. Exact values come from running the artifact on a board.

<a id="entry-points"></a>
## Entry Points

- Model preparation: [`model/README.md`](model/README.md) — eight manifest-backed HBM assets for both S100 and S100P.
- Python runtime: [`runtime/python/README.md`](runtime/python/README.md) — preprocessing, metadata binding, selected-submodel execution, and JSON summary CLI.
- C++ runtime: not provided; C++ is `not-supported`.
- Conversion: [`conversion/README.md`](conversion/README.md) — source model identity, toolchain and targets, export/calibration/compile steps, and post-conversion validation.
- Evaluation: [`evaluator/README.md`](evaluator/README.md) — source-recorded tables and the array-capture comparison procedure.

<a id="license"></a>
## License

Sample code is Apache-2.0 under the repository [LICENSE](../../../LICENSE). The manifest points to precompiled Google-origin SigLIP artifacts; their individual weight/export license and version were not recorded in the source release, so no additional model license is asserted here. Preserve the source contributor attribution: Cauchy @吴超.
