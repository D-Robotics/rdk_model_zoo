English | [简体中文](./README_cn.md)

# CLIP image-text matching

<a id="overview"></a>
## Overview

CLIP maps an image and candidate texts into a shared 512-dimensional space and ranks the texts by cosine similarity. This sample keeps the source asset boundary: the image encoder is an X5 BPU `.bin` model and the text encoder is a CPU ONNX model, with the source BPE vocabulary preserved in `runtime/python/bpe_simple_vocab_16e6.txt.gz`. The source commit is `ac115717197920355fc390bb04299b20e6436864`.

The maintained task has three stages: `pre_process` converts one BGR image and prompt list into image/tokens tensors, `forward` runs both encoders and returns raw features, and `post_process` computes cosine scores and descending order. It also provides `predict`; visualization remains a separate helper.

<a id="support-matrix"></a>
## Support Matrix

The only published pair is for RDK X5. No S100, S100P, or S600 CLIP pair is present in the active manifest, and no C++ runtime is provided. Board status is not a board test claim.

| Variant | x5 | s100 | s100p | s600 | Python | C++ |
| --- | --- | --- | --- | --- | --- | --- |
| `clip-image-text-pair` | supported-not-run | not-supported | not-supported | not-supported | supported-not-run | not-supported |

Board verification evidence: not-run. Host tests use injected image/ONNX fixtures and source tokenizer parity; they do not certify X5 execution.

<a id="prerequisites"></a>
## Prerequisites

- Board: RDK X5 image providing `hbm_runtime` for the image encoder and `onnxruntime` for the CPU text encoder. Image and firmware versions were not verified.
- Host checks: Python 3.14.7 with `numpy`, `opencv-python`, `PyYAML`, `ftfy==6.3.1`, and `regex==2026.9.10`; host tests do not require ONNX Runtime.
- Board inference additionally requires `onnxruntime`, the two model assets, and the bundled BPE vocabulary.

<a id="quickstart"></a>
## Quick Start

Prepare both models explicitly, then run on X5 from the repository root. The current `run.sh` does not download automatically; this is an intentional difference from the historical source launcher.

```bash
# cwd: repository root; source: exact URLs in docs/release/x5/models.yaml
python3 samples/vision/clip/model/download.py --target x5
# expect: samples/vision/clip/model/img_encoder.bin and text_encoder.onnx

# cwd: repository root; input: test_data/dog.jpg; prompts: default a diagram,a dog
python3 samples/vision/clip/runtime/python/main.py --target x5
# expect: JSON prompts/scores/order and annotated samples/vision/clip/test_data/inference.png; exit code 0
```

The default visualization path is the tracked `test_data/inference.png` and will be overwritten. Pass `--img-save-path` to preserve it elsewhere. `run.sh` is a convenience launcher only; it does not prepare models.

<a id="expected-results"></a>
## Expected Results

The CLI prints `target`, `prompts`, `scores`, `order`, and `image_saved`. `scores` are cosine similarities in prompt order; `order` contains descending prompt indices. The visualization writes each prompt and score onto a copy of the input image. The source evaluator expects the dog image to score higher for `a dog` than for `a diagram`; no numeric benchmark is published.

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
- Evaluation: [`evaluator/README.md`](evaluator/README.md) — source validation path and no fabricated benchmark values.

<a id="license"></a>
## License

Sample code follows the repository [LICENSE](../../../LICENSE), Apache-2.0. The source CLIP model assets retain the licenses and provenance recorded by the X5 publication; this sample does not add a new model license assertion. Source contributor attribution is preserved through the source migration record.
