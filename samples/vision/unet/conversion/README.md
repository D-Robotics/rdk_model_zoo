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

<a id="source-model"></a>
## Source model

All five backbones share onnx_export/model. Source attribution, pinned reference commit and MIT terms are in the [sample README](../README.md#license). Supply a checkpoint trained for the chosen backbone; this repository does not download or fabricate it. Do not bypass failed strict loading with strict=False.

<a id="toolchain-targets"></a>
## Toolchain

Target: X5 / bayes-e. The source does not pin an OE Docker version. Use x86 Linux with hb_mapper and hb_model_info; mapper records actual versions. Export additionally needs PyTorch, ONNX, ONNX Runtime and NumPy; the calibration example needs Pillow and mapper needs PyYAML. OE was not installed/executed this round, and untested version combinations are not certified.

<a id="export"></a>
## 1. Export ONNX

Use a checkpoint trained for the selected backbone. The exporter performs a
strict load, writes a fixed opset-11 graph, runs ONNX checker, and compares the
same deterministic input with ONNX Runtime. Existing output and report files
are never overwritten.

```bash
# cwd: samples/vision/unet/conversion
python3 onnx_export/export_unet.py \
  --backbone resnet18 \
  --checkpoint /models/unet_resnet18_voc_best.pth \
  --output /models/unet_resnet18_voc_512x512.onnx
```

The numerical check must pass before `mapper.py` accepts the export report.
`--skip-runtime-check` is only a structural preflight and deliberately marks
the ONNX as not ready for X5 PTQ.

<a id="calibration"></a>
## 2. Prepare calibration tensors

Use about 100 representative Pascal VOC training images. Each calibration
file must be a headerless, little-endian float32 `.bin` containing one RGB CHW
tensor with shape `[3, 512, 512]` and values in `[0, 255]`. Do not divide by
255 in the data script: the PTQ YAML owns normalization through
`data_scale=1/255` so that the same rule is applied to the Runtime NV12 input.

`mapper.py` reads every tensor, rejects wrong sizes, NaN/Inf and out-of-range
values, and writes a hashed `calibration-manifest.json` into the run reports.

```python
# cwd: samples/vision/unet/conversion; prepare a representative VOC image folder first
from pathlib import Path
import numpy as np
from PIL import Image

images = sorted(Path("/data/VOC2012/JPEGImages").glob("*.jpg"))[:100]
if not images:
    raise ValueError("No calibration images found")
out = Path("/data/unet/calibration_data_rgb_f32_512")
out.mkdir(parents=True, exist_ok=False)
for source in images:
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR))
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype="<f4")
    with (out / (source.stem + ".bin")).open("xb") as stream:
        stream.write(chw.tobytes())
print(len(images), "calibration tensors", out)
```

<a id="compile"></a>
## 3. Compile for X5

Run inside an OpenExplorer Mapper environment that provides `hb_mapper` and
`hb_model_info`. The output directory must not exist.

```bash
# cwd: samples/vision/unet/conversion
python3 mapper.py \
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

<a id="validation"></a>
## Validation

Default export compares PyTorch/ORT on deterministic input. skip-runtime-check is structural preflight only and cannot pass the later PTQ gate. Evaluate compiled artifacts on the same VOC manifest through evaluator --model; newly compiled files must not masquerade as publisher-hash artifacts.

```bash
# cwd: repository root; on X5 after a real successful compile, with a prepared VOC manifest
python3 samples/vision/unet/evaluator/eval_unet.py --backend x5 --backbone resnet18 --model /output/unet_resnet18_x5_run_001/artifacts/unet_resnet18_voc_512x512_nv12.bin --manifest /data/unet/val.tsv --report /reports/unet_custom_x5.json --min-miou 0.50
```

Use the actual BIN path recorded in run-receipt.json; replace --model above with that receipt path. Current unified-entry board tests, full export, compilation and dataset evaluation are not-run.

<a id="artifacts"></a>
## Artifacts

Outputs retain ONNX/report, resolved YAML, calibration manifest, checker/build/model-info logs, artifacts/hashes and run-receipt.json. See the [model table](../model/README.md#artifacts) for the five published BINs and hashes; do not assign their hashes to recompiles.

<a id="known-gaps"></a>
## Known gaps

Supply trained checkpoints, full VOC and a representative calibration subset. Source OE image/framework versions are not pinned. This round validates host logic, entrypoints and documentation, not conversion/accuracy. Taking the first 100 files demonstrates format preparation, not calibration representativeness; choose a subset suitable for deployment data.
