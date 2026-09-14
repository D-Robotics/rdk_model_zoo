[English](./README.md) | [简体中文](./README_cn.md)

# UNet conversion for RDK X5

This directory contains only the float-checkpoint-to-X5 conversion path for the
five UNet ResNet variants. Training remains outside the Model Zoo repository.

## Layout

```text
conversion/
├── mapper.py
├── onnx_export/
│   ├── export_unet.py
│   └── model/
└── ptq_yamls/
```

`onnx_export/model` is the single source of the PyTorch architecture used by
checkpoint training, evaluation, and export. `ptq_yamls` contains one reviewed
`bayes-e` template per backbone. `mapper.py` binds a template to one ONNX,
calibration set, and new output directory before it runs checker, makertbin, and
`hb_model_info`.

## 1. Export ONNX

Use a checkpoint trained for the selected backbone. The exporter performs a
strict load, writes a fixed opset-11 graph, runs ONNX checker, and compares the
same deterministic input with ONNX Runtime. Existing output and report files
are never overwritten.

```bash
python onnx_export/export_unet.py \
  --backbone resnet18 \
  --checkpoint /models/unet_resnet18_voc_best.pth \
  --output /models/unet_resnet18_voc_512x512.onnx
```

The numerical check must pass before `mapper.py` accepts the export report.
`--skip-runtime-check` is only a structural preflight and deliberately marks
the ONNX as not ready for X5 PTQ.

## 2. Prepare calibration tensors

Use about 100 representative Pascal VOC training images. Each calibration
file must be a headerless, little-endian float32 `.bin` containing one RGB CHW
tensor with shape `[3, 512, 512]` and values in `[0, 255]`. Do not divide by
255 in the data script: the PTQ YAML owns normalization through
`data_scale=1/255` so that the same rule is applied to the Runtime NV12 input.

`mapper.py` reads every tensor, rejects wrong sizes, NaN/Inf and out-of-range
values, and writes a hashed `calibration-manifest.json` into the run reports.

## 3. Compile for X5

Run inside an OpenExplorer Mapper environment that provides `hb_mapper` and
`hb_model_info`. The output directory must not exist.

```bash
python mapper.py \
  --backbone resnet18 \
  --onnx /models/unet_resnet18_voc_512x512.onnx \
  --calibration /data/unet/calibration_data_rgb_f32_512 \
  --output /output/unet_resnet18_x5_run_001
```

The guarded sequence is:

```text
export report → calibration audit → hb_mapper checker → hb_mapper makertbin
              → exactly one .bin → hb_model_info BPU march: bayes-e
```

The run keeps the resolved YAML, checker/build/model-info logs, calibration
manifest, copied artifacts, hashes, tool versions, and `run-receipt.json`.
A successful compile is still followed by accuracy evaluation and board
Runtime verification with `../evaluator/eval_unet.py`.
