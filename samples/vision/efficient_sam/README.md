English | [简体中文](README_cn.md)

# EfficientSAM-Tiny

EfficientSAM-Tiny segments any object from a single box prompt, using a distilled
ViT-Tiny image **encoder** and a fixed-prompt mask **decoder** compiled as two
separate `.hbm` models for the RDK-S series (S100 / S100P / S600). Board
inference uses `hbm_runtime`.

## Algorithm overview

EfficientSAM distills Segment Anything into a ViT-Tiny backbone. The encoder maps
a normalized RGB 512×512 image to a 256×32×32 embedding; the decoder takes that
embedding (the box prompt is baked into the exported decoder ONNX) and predicts a
low-resolution mask plus an IoU score. The runtime upsamples the selected mask to
512×512 and overlays it.

- Paper: <https://arxiv.org/abs/2312.00863>
- Project website: <https://yformer.github.io/efficient-sam/>
- Official repository: <https://github.com/yformer/EfficientSAM>

## Capabilities

- Single-box prompt → binary object mask + IoU (prompt baked into decoder ONNX).

## Platform compatibility

| Board | SoC | march | Model dir |
|---|---|---|---|
| S100 | s100 | nash-e | `model/nash-e/` |
| S100P | s100p | nash-m | `model/nash-m/` |
| S600 | s600 | nash-p | `model/nash-p/` |

## Directory structure

```
efficient_sam/
├── conversion/          # ONNX export + hb_compile quantization
│   ├── configs/         # one committed YAML per march (encoder + decoder)
│   └── scripts/         # quantize.py, export/prepare_*.py
├── evaluator/           # board-side numerical evaluation notes
├── model/               # download_model.sh + per-march .hbm
├── runtime/python/      # hbm_runtime inference: main.py, efficient_sam.py, run.sh
└── test_data/           # dogs.jpg + expected binary mask
```

## Quick start

On the board:

```bash
cd samples/vision/efficient_sam/runtime/python
bash run.sh
# -> writes test_data/efficient_sam_full_mask_result.jpg + efficient_sam_binary_mask_result.png
```

## Conversion

See [`conversion/README.md`](./conversion/README.md) for ONNX export,
quantization configs, and OE toolchain entry points.

## Runtime

See [`runtime/python/README.md`](./runtime/python/README.md).

## Evaluation

Board-side latency measurements are in
[`evaluator/README.md`](./evaluator/README.md).

## License

This sample follows the RDK Model Zoo license. Upstream EfficientSAM weights and
ONNX assets retain their original license.
