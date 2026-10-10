English | [简体中文](README_cn.md)

# Pinned MobileNet host workflow

Use this workflow to fetch an exact timm checkpoint, export batch-one FP32
ONNX, prepare target-specific calibration, and evaluate a complete labeled
dataset. Each sample exposes `conversion/export.py` and `evaluator/evaluate.py`.
Run the commands below from the repository root; choose a new output directory
for every run. Model files and datasets belong outside the repository.

## Models and environment

`checkpoints.json` pins six checkpoints by repository revision and file SHA256:
`v1-100`, `v2-100`, `v3-large-100`, `v4-small`, `v4-medium-224`, and
`v4-medium-256`. Medium-224 targets all four platforms; Medium-256 targets S100/S100P/S600.
Other variants have all four targets in their candidate matrix. A matrix
entry is not a claim that a compiled model has passed board acceptance.

Use Python 3.10 with the versions in `requirements.lock.txt`. Install the
PyTorch 2.8.0 CUDA 12.8 wheels using the
[official PyTorch instructions](https://pytorch.org/get-started/previous-versions/#v280),
then install this lock. CPU ONNX evaluation works in the same environment.
Keep a complete `python -m pip freeze` with each campaign; the local validated
environment also records the inherited base environment and wheel install report.

```bash
python -m pip install -r utils/tools/mobilenet/requirements.lock.txt
```

## Source and preprocessing

The pins include model-card URLs, licenses, upstream preprocessing settings,
and the explicit deployment contract. V1 uses mean/std 0.5; the other models
use their recorded ImageNet normalization. Each model uses PIL bicubic
shorter-edge resize followed by center crop. `crop_pct` and the deployed size
are explicit; upstream `test_input_size` does not override them.

All new graphs take normalized RGB float32 NCHW `[1,3,H,W]` and return logits
`[1,1000]`. Classes follow timm `ImageNetInfo(ilsvrc2012)` indices 0–999.
These are new checkpoint builds; historical manifest artifacts can have a
different source, preprocessing, tensor shape, and probability output.

```bash
python utils/tools/mobilenet/workflow.py fetch --model v4-small \
  --output /absolute/workbench/weights/v4-small

python samples/vision/mobilenetv4/conversion/export.py \
  --model v4-small --source-dir /absolute/workbench/weights/v4-small \
  --images samples/vision/mobilenetv4/test_data/great_grey_owl.JPEG \
           samples/vision/mobilenetv4/test_data/zebra_cls.jpg \
  --opset 11 --simplify --output /absolute/workbench/run/export
```

Export checks the source hashes, strictly loads safetensors without network
fallback, validates ONNX, and compares real-image logits and Top-5 against
PyTorch. `export.json` binds the graph hash, geometry, dependency versions,
source hashes, actual parameter count, and numerical comparisons. Reported
Conv/Linear MACs have an explicit limited scope and are not total GFLOPs.

## Dataset manifests and input freeze

A labeled manifest has `dataset`, `count`, `class_count: 1000`, and an ordered
`images` array of `{path, sha256, label_id}` records. Labels are numeric model
class IDs, not lexicographic folder positions. Calibration records can use
`name` instead of `path` and omit labels. Every image hash is checked.

For the ImageNetV2 10,000-image / COCO 200-image campaign, `freeze.py` verifies
all sources, all images, ten images per evaluation class, and disjoint hashes.
It copies manifests, checkpoint pins, a full dependency lock, toolchain image
identity, and the label mapping into a new campaign directory with SHA256SUMS.

```bash
python utils/tools/mobilenet/freeze.py \
  --source-root /absolute/workbench/inputs/source_models/mobilenet \
  --evaluation-manifest /absolute/workbench/imagenetv2/manifest.json \
  --evaluation-root /absolute/workbench/imagenetv2/images \
  --calibration-manifest /absolute/workbench/calibration/manifest.json \
  --calibration-root /absolute/workbench/calibration \
  --environment-lock /absolute/workbench/environment.lock.txt \
  --toolchains /absolute/workbench/toolchains.json \
  --output /absolute/workbench/campaign
```

The source root layout is `<repo-name>/<revision>/{config.json,README.md,model.safetensors}`.
`toolchains.json` records the actual Docker tags and immutable image IDs.
The campaign's one-percentage-point Top-1/Top-5 drop limits are local candidate
gates, not a repository-wide policy. Changing a frozen input creates a new
campaign. COCO calibration suitability is checked by subsequent quantized
evaluation on the complete independent classification dataset.

## Calibration and OE configuration

The calibration writer shares the exact RGB geometry with evaluation. For
X5 OE v1.2.8 it writes raw RGB float32 NCHW in [0,255]; the OE loader performs
the RGB/NV12 transformation. For S OE v3.7.0 it writes normalized ONNX input
float32 NCHW. X5 files are headerless little-endian float32 `.rgb`; S files
are `.npy`. The X5 loader does not support NumPy file headers.
Both YAMLs describe RGB training input and NV12 runtime input,
with mean ×255 and scale 1/(255×std). Do not normalize the X5 data twice.

```bash
python utils/tools/mobilenet/workflow.py calibrate --model v4-small \
  --platform x5 --manifest /absolute/workbench/calibration/manifest.json \
  --images-root /absolute/workbench/calibration --expected-images 200 \
  --evaluation-manifest /absolute/workbench/imagenetv2/manifest.json \
  --output /absolute/workbench/run/x5/calibration

python utils/tools/mobilenet/workflow.py config --model v4-small \
  --platform x5 --export-dir /absolute/workbench/run/export \
  --calibration-dir /absolute/workbench/run/x5/calibration \
  --output /absolute/workbench/run/x5/config
```

Repeat both commands with `s100` and new output paths for the S trial.
Generated configuration receipts include the target march and compilation
command. Mount host input/output paths at the same absolute paths in Docker.
Inside the matching pinned container, check configuration and all 200 loader
outputs without compiling:

```bash
python3 /absolute/repository/utils/tools/mobilenet/check_toolchain.py \
  --platform x5 --config /absolute/workbench/run/x5/config/config.yaml
```

The calibration batch is the toolchain default; determine its actual value
from compilation logs. This workflow does not apply YOLO graph rewrites.

## Complete FP32 ONNX evaluation

```bash
python samples/vision/mobilenetv4/evaluator/evaluate.py \
  --model v4-small --export-dir /absolute/workbench/run/export \
  --manifest /absolute/workbench/imagenetv2/manifest.json \
  --images-root /absolute/workbench/imagenetv2/images --expected-images 10000 \
  --provider CPUExecutionProvider --threads 4 \
  --output /absolute/workbench/run/evaluation
```

The evaluator checks the full manifest and the export identity, loads ONNX
once, writes each prediction, and reports Top-1/Top-5 in [0,1]. The receipt
records the actual ORT provider; a missing requested CUDA provider is an error.
This RGB FP32 baseline does not include an NV12 roundtrip. Subsequent board
evaluation should apply `prepare_rgb`, pass its contiguous BGR view to the
sample classifier with `resize_type=0` at the same size, and record NV12 and
quantization effects explicitly. Do not use the legacy default letterbox for
this checkpoint contract.

## Host verification

```bash
python -m unittest utils.py_utils.tests.test_classification_host -v
python -m unittest discover -s utils/tools/mobilenet/tests -v
```

These checks cover timm preprocessing parity, normalization domains, labels,
counts, overlap, output ranking, and identity mismatches. They do not establish
Runtime/C++ performance or released status.
