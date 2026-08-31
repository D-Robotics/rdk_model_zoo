English | [简体中文](./README_cn.md)

# DINOv2 Model Description

DINOv2 is a self-supervised vision transformer backbone published by Meta AI
(Oquab et al., 2023). This sample deploys the ViT-S/14 variant, quantized to
int16, on RDK S100/S100P/S600 as a general-purpose image embedding model. It
is the first self-supervised foundation-model sample in the RDK model zoo.

## Algorithm Overview

DINOv2 is a plain ViT: a patch-14 convolutional stem, 12 pre-LN transformer
blocks (fused-qkv attention + GELU MLP + LayerScale), and a final LayerNorm.
The deployed graph rewrites the SDPA attention into explicit MatMul + Softmax
and bakes the interpolated positional embedding as a constant, so the whole
network compiles to BPU operators with zero CPU fallback.

## Algorithm Capabilities

- Global image embedding: `cls_feat`, shape `(1, 384)`.
- Dense per-patch features: `patch_feat`, shape `(1, 256, 384)`.
- Downstream uses: image retrieval / similarity, linear-probe classification,
  and as a frozen backbone for dense tasks (the same encoder family powers
  Depth Anything V2).

## Algorithm Features

- int16 PTQ with default (KL) calibration: end-to-end output cosine 0.998+ on
  board (see [evaluator](./evaluator/README.md)).
- 100% BPU execution, no CPU-op fallback (800/800 operators on nash-e).
- One `.hbm` per march (Nash-E / Nash-M / Nash-P), auto-selected on board.

## Directory Structure

```text
dinov2/
├── conversion                     # ONNX export + PTQ conversion pipeline
│   ├── onnx_export/               # PyTorch -> ONNX export script
│   └── mapper.py                  # One-command conversion entry
├── evaluator                      # Measured performance / accuracy records
├── model                          # download_model.sh + model list
├── runtime                        # Board inference demo
│   └── python                     # hbm_runtime based python runtime
└── test_data                      # Demo images
```

## Quick Start

```bash
# On the board:
cd samples/vision/dinov2/runtime/python
bash run.sh
```

The script downloads the `.hbm` matching the on-board SoC, runs inference on
the two test images, prints output summaries, and reports the cosine
similarity between the two image embeddings.

## Model Conversion

See [conversion/README.md](./conversion/README.md) for the quantization
recipe and the measured configuration matrix.

## Runtime

See [runtime/python/README.md](./runtime/python/README.md) for the input
contract, API usage, and CLI options.

## Model Evaluation

See [evaluator/README.md](./evaluator/README.md) for measured latency,
PTQ/board cosine, and reproduction commands.

## Model List

| Model Name | Input Size | Embedding (cls / patch) | Params | RDK S100 | RDK S100P | RDK S600 |
|---|---|---|---|---|---|---|
| dinov2_vits14_224_int16 | 1x3x224x224 | (1,384) / (1,256,384) | 22.06 M | 3.73 ms / 267.44 FPS | 3.02 ms / 329.53 FPS | 2.25 ms / 441.64 FPS |

## Contributors

D-Robotics model zoo team.

## License

The source model and weights are Apache-2.0 licensed
[DINOv2](https://github.com/facebookresearch/dinov2) artifacts. See
[../../../LICENSE](../../../LICENSE).
