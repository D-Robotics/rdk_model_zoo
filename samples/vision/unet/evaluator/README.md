[English](./README.md) | [简体中文](./README_cn.md)

# UNet evaluator

`eval_unet.py` is the single accuracy entry point for the five UNet ResNet
variants. It evaluates the same Pascal VOC path-pair manifest with a PyTorch
checkpoint, a float ONNX model, or an RDK X5 `bayes-e` `.bin` model.

The evaluator resizes RGB images and class-index masks to `512x512`, preserves
the VOC ignore label `255`, applies `argmax` to the 21-class logits, and reports
mIoU, pixel accuracy, and per-class IoU. Every run writes model and manifest
hashes to a new JSON report.

<a id="dataset"></a>
## Manifest

Each non-empty line contains an absolute image path and mask path separated by
one tab:

```text
/data/VOC2012/JPEGImages/2007_000033.jpg\t/data/VOC2012/SegmentationClass/2007_000033.png
```

VOC palette masks are read as class indices; they must not be converted to
grayscale before evaluation.

<a id="environment"></a>
## Environment

Common requirements: Python 3.10+, NumPy, Pillow. PyTorch/ONNX backends additionally need torch/onnxruntime respectively. X5 also needs OpenCV, PyYAML, matching SDK and verifies actual X5 identity/OS; it reuses the canonical runtime stages. Floating backends preserve source float-input rules.

| Parameter | Default | Meaning |
| --- | --- | --- |
| --model | required | .pth / .onnx / .bin |
| --manifest | required | image TAB mask absolute path pairs |
| --report | required | new JSON path; existing file rejected |
| --backend | auto | suffix detection or pytorch/onnx/x5 |
| --backbone | None | required for .pth or custom-named .bin |
| --limit | None | first N manifest entries; positive |
| --progress-every | 50 | progress interval |
| --min-miou | 0.0 | write report and return 2 if below threshold |

Standard published filenames identify the backbone; custom compiled filenames require --backbone, and conflicting filename/backbone pairs are rejected. --model is explicit evaluation input, not publisher-hash authentication. Reports record its actual digest; this path must not be presented as a verified download.

<a id="command"></a>
## PyTorch checkpoint

Run this backend on the development machine with PyTorch and Pillow installed:

```bash
# cwd: samples/vision/unet/evaluator
python3 eval_unet.py \
  --model /models/unet_resnet18_voc_best.pth \
  --backbone resnet18 \
  --manifest /data/unet/val.tsv \
  --report /reports/unet_resnet18_pytorch.json \
  --min-miou 0.50
```

## ONNX

Install `onnxruntime` in the host evaluation environment, then run:

```bash
# cwd: samples/vision/unet/evaluator
python3 eval_unet.py \
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
# cwd: samples/vision/unet/evaluator
python3 eval_unet.py \
  --model /models/unet_resnet18_voc_512x512_nv12.bin \
  --manifest /data/unet/val.tsv \
  --report /reports/unet_resnet18_x5.json \
  --min-miou 0.50
```

Use `--limit` for a smoke run. A report below `--min-miou` is still written,
but the command exits with status 2.

<a id="metrics"></a>
## Metrics

mIoU averages classes with union>0; pixel accuracy counts correct nonignored pixels; class_iou contains 21 class IoUs. Label 255 is ignored. Images/masks resize to 512×512 (RGB bilinear, label nearest). Do not compare directly with original-resolution evaluation or substitute a --limit subset for the historical full 1449-image set.

<a id="outputs"></a>
## Outputs

A new JSON stores backend, model/manifest hashes, sample count and input_contract. runtime holds environment/model metadata; metrics contains miou, pixel_accuracy and class_iou. min_miou, passed and elapsed_seconds are also recorded. The internal confusion matrix is not serialized. Falling below the threshold still writes the report and returns 2, so file existence alone is not success.

<a id="reference-results"></a>
## Reference results

The [sample README](../README.md) retains five-backbone training/PTQ tables and the 1449-image ResNet18 three-backend record. These retain their original checkpoint/artifact conditions. This round did not remeasure mIoU, BPU latency or FPS; synthetic host tests do not replace that evidence.

<a id="boundaries"></a>
## Boundaries

Prepare full datasets and checkpoint/ONNX/BIN beforehand. Full dataset evaluation and board execution are not-run. Runtime timing is not a pure BPU benchmark supplied by this evaluator. Strict PyTorch checkpoint loading and single-input/output ONNX constraints remain; arbitrary architectures or S-family assets are not supported.
