English | [简体中文](./README_cn.md)

# CLIP image-text matching

<a id="overview"></a>
## Overview

CLIP maps an image and candidate texts into a shared 512-dimensional space and ranks the texts by cosine similarity. The image encoder is an X5 BPU `.bin` model and the text encoder is a CPU ONNX model; the BPE vocabulary ships as `runtime/python/bpe_simple_vocab_16e6.txt.gz`. Source: X5 platform sample delivery at `ac115717197920355fc390bb04299b20e6436864`.

The `CLIPTask` has three stages: `preprocess` converts one BGR image and prompt list into image/tokens tensors, `infer` runs both encoders and returns raw features, and `postprocess` computes cosine scores and descending order. `predict` chains the three stages; visualization remains a separate helper.

<a id="support-matrix"></a>
## Support Matrix

The only published pair is for RDK X5. No S100, S100P, or S600 CLIP pair is present in the active manifest, and no C++ runtime is provided.

| Variant | x5 | s100 | s100p | s600 | Python | C++ |
| --- | --- | --- | --- | --- | --- | --- |
| `clip-image-text-pair` | supported | not-supported | not-supported | not-supported | supported | not-supported |

Board execution requires an X5 board with `hbm_runtime` and `onnxruntime`; host tests use injected image/ONNX fixtures and do not execute the BPU model.

<a id="prerequisites"></a>
## Prerequisites

- Board: RDK X5 image providing `hbm_runtime` for the image encoder and `onnxruntime` for the CPU text encoder. No specific image or firmware version is pinned.
- Host checks: Python 3.14.7 with `numpy`, `opencv-python`, `PyYAML`, `ftfy==6.3.1`, and `regex==2026.9.10`; host tests do not require ONNX Runtime.
- Board inference additionally requires `onnxruntime`, the two model assets, and the bundled BPE vocabulary.

<a id="quickstart"></a>
## Quick Start

Prepare both models explicitly, then run on X5 from the repository root. `run.sh` is a convenience launcher and does not download models automatically.

```bash
# cwd: repository root; source: exact URLs in docs/release/x5/models.yaml
python3 samples/vision/clip/model/download.py --target x5
# expect: samples/vision/clip/model/img_encoder.bin and text_encoder.onnx

# cwd: repository root; input: test_data/dog.jpg; prompts: default a diagram,a dog
python3 samples/vision/clip/runtime/python/main.py --target x5
# expect: JSON prompts/scores/order and annotated samples/vision/clip/test_data/inference.png; exit code 0
```

The default visualization path is the tracked `test_data/inference.png` and is overwritten by each run. Pass `--img-save-path` to write the image elsewhere. `run.sh` is a convenience launcher only; it does not prepare models.

<a id="expected-results"></a>
## Expected Results

The CLI prints `target`, `prompts`, `scores`, `order`, and `image_saved`. `scores` are cosine similarities in prompt order; `order` contains descending prompt indices. The visualization writes each prompt and score onto a copy of the input image. The expected qualitative result for `dog.jpg` is a higher score for `a dog` than for `a diagram` (source validation expectation); no numeric benchmark is published.

<a id="directory"></a>
## Directory Layout

```text
clip/
├── conversion/             # image/text protocol and conversion boundary
├── evaluator/              # validation conditions; no published benchmark
├── model/                  # paired manifest-backed model preparation
├── runtime/python/         # BPE, preprocessing, dual encoder runner, task, CLI, drawing
├── test_data/              # dog.jpg and inference.png
└── README.md               # this guide
```

<a id="entry-points"></a>
## Entry Points

- Model preparation: [`model/README.md`](model/README.md) — X5 image `.bin` and text `.onnx` pair with separate asset identities.
- Python runtime: [`runtime/python/README.md`](runtime/python/README.md) — BPE tokenization, BPU image + CPU ONNX text inference, cosine ranking, and visualization.
- C++ runtime: not provided; C++ is `not-supported`.
- Conversion: [`conversion/README.md`](conversion/README.md) — source protocol and missing export/calibration recipe.
- Evaluation: [`evaluator/README.md`](evaluator/README.md) — validation commands and the qualitative expectation; no published benchmark values.

<a id="license"></a>
## License

Sample code follows the repository [LICENSE](../../../LICENSE), Apache-2.0. The CLIP model assets retain the licenses and provenance recorded by the X5 publication; no additional model license is asserted here. Contributor attribution from the X5 source delivery is preserved.
