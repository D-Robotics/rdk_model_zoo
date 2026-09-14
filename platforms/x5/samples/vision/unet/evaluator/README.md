[English](./README.md) | [简体中文](./README_cn.md)

# UNet evaluator

`eval_unet.py` is the single accuracy entry point for the five UNet ResNet
variants. It evaluates the same Pascal VOC path-pair manifest with a PyTorch
checkpoint, a float ONNX model, or an RDK X5 `bayes-e` `.bin` model.

The evaluator resizes RGB images and class-index masks to `512x512`, preserves
the VOC ignore label `255`, applies `argmax` to the 21-class logits, and reports
mIoU, pixel accuracy, and per-class IoU. Every run writes model and manifest
hashes to a new JSON report.

## Manifest

Each non-empty line contains an absolute image path and mask path separated by
one tab:

```text
/data/VOC2012/JPEGImages/2007_000033.jpg\t/data/VOC2012/SegmentationClass/2007_000033.png
```

VOC palette masks are read as class indices; they must not be converted to
grayscale before evaluation.

## PyTorch checkpoint

Run this backend on the development machine with PyTorch and Pillow installed:

```bash
python eval_unet.py \
  --model /models/unet_resnet18_voc_best.pth \
  --backbone resnet18 \
  --manifest /data/unet/val.tsv \
  --report /reports/unet_resnet18_pytorch.json \
  --min-miou 0.50
```

## ONNX

Install `onnxruntime` in the host evaluation environment, then run:

```bash
python eval_unet.py \
  --model /models/unet_resnet18_voc_512x512.onnx \
  --manifest /data/unet/val.tsv \
  --report /reports/unet_resnet18_onnx.json \
  --min-miou 0.50
```

## RDK X5 binary

The `.bin` backend must run locally on an aarch64 RDK X5 with RDK OS 3.5.0 or
newer. Use the X5 `hbm_runtime` package shipped with the board image; do not
install a same-named package for another platform from PyPI. The compiled model
must expose one packed NV12 input and one 21-class logits output.

```bash
python eval_unet.py \
  --model /models/unet_resnet18_voc_512x512_nv12.bin \
  --manifest /data/unet/val.tsv \
  --report /reports/unet_resnet18_x5.json \
  --min-miou 0.50
```

Use `--limit` for a smoke run. A report below `--min-miou` is still written,
but the command exits with status 2.
