English | [简体中文](README_cn.md)

# MobileSAM

MobileSAM segments any object in an image from a single box prompt. This sample
compiles the TinyViT image **encoder** and the box-prompt mask **decoder** as two
separate `.hbm` models for the RDK-S series (S100 / S100P / S600) and runs
full-mask inference on board with `hbm_runtime`.

## Algorithm overview

MobileSAM distills Segment Anything into a lightweight TinyViT backbone so the
encoder runs in real time on edge BPU. The encoder maps a normalized 512×512
image to a 256×32×32 embedding; the decoder takes that embedding plus one box
prompt `(x1, y1, x2, y2)` and predicts a low-resolution mask plus an IoU score.
The runtime upsamples the selected mask back to 512×512 and overlays it.

- Paper: <https://arxiv.org/abs/2306.14289>
- Official repository: <https://github.com/ChaoningZhang/MobileSAM>

## Capabilities

- Single-box prompt → binary object mask + IoU.

## Platform compatibility

| Board | SoC | march | Model dir |
|---|---|---|---|
| S100 | s100 | nash-e | `model/nash-e/` |
| S100P | s100p | nash-m | `model/nash-m/` |
| S600 | s600 | nash-p | `model/nash-p/` |

The board is auto-detected at runtime from `/sys/class/boardinfo/`; pick a march
explicitly with `download_model.sh <march>` or `quantize.py --march <march>`.

## Directory structure

```
mobile_sam/
├── conversion/          # ONNX export + hb_compile quantization
│   ├── configs/         # one committed YAML per march (encoder + decoder)
│   └── scripts/         # quantize.py, export/prepare_*.py
├── evaluator/           # board-side numerical evaluation notes
├── model/               # download_model.sh + per-march .hbm
├── runtime/python/      # hbm_runtime inference: main.py, mobile_sam.py, run.sh
└── test_data/           # dogs.jpg + expected binary mask
```

## Quick start

On the board:

```bash
cd samples/vision/mobile_sam/runtime/python
bash run.sh
# -> writes test_data/mobile_sam_full_mask_result.jpg + mobile_sam_binary_mask_result.png
```

`run.sh` auto-detects the board, downloads the matching `.hbm` pair if missing,
then runs `python3 main.py`.

## Conversion

See [`conversion/README.md`](./conversion/README.md) for ONNX export,
quantization configs, and OE toolchain entry points.

## Runtime

See [`runtime/python/README.md`](./runtime/python/README.md). Entry point
`main.py` resolves the board, loads both `.hbm` models, runs the encoder then
the decoder, and saves the overlay + binary mask.

## Evaluation

See [`evaluator/README.md`](./evaluator/README.md) for board-side latency
measurements.

## License

This sample follows the RDK Model Zoo license. The upstream MobileSAM weights
and ONNX assets retain their original license.
