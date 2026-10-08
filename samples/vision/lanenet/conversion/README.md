[English](README.md) | [简体中文](README_cn.md)

# LaneNet conversion

This directory provides compiler YAML, calibration preparation and a compilation
helper. Prepare the matching model/export code described below to generate ONNX;
the calibration and compilation commands then consume that ONNX.

<a id="source-model"></a>
## Source model and prerequisites

The source checkpoint URL is
[best_model.pth](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Lanenet/best_model.pth).
Obtain the matching LaneNet model/export code and its `test.py` export entry,
then record the checkpoint checksum, code revision and architecture. The export
entry uses `test.py --img... --model best_model.pth`; configure its graph and
preprocessing to match the runtime contract below.

Export dependencies are Python≥3.6, Torch≥1.2, torchvision≥0.4.0, NumPy≥1.7,
OpenCV, pandas and matplotlib. Record the exact installed versions for your
export. Calibration/configuration tools use Python, NumPy, OpenCV and PyYAML.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── compile.py  # Python script
├── config.yaml  # Configuration
└── prepare_calibration.py  # Python script
```

<a id="toolchain-targets"></a>
## Toolchain and target

`config.yaml` declares **nash-e/S100**, latency mode, O2 and
`set_all_nodes_int16`. Choose the matching OE Docker/compiler environment using [OE environment](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)
and [toolchain manual](https://toolchain.d-robotics.cc/). Prepare the matching x86
Linux toolchain explicitly. The helper never installs it.

The published model targets S100. For S100P/S600, prepare a graph, march/configuration
and runtime binding for that target. Inspect the compiled public embedding/binary
tensor dtypes separately from the internal int16 quantization setting.

<a id="export"></a>
## Export contract and configuration

Supply the exported nonempty ONNX. Its input is float32 RGB NCHW `[1,3,256,512]`, INTER_AREA stretch, `/255` then
ImageNet mean/std. Runtime consumes named float32 `instance_seg_logits` and int64
`binary_seg_pred`. Inspect the compiled output metadata and retain any auxiliary
tensors with their actual names, shapes and types.

The YAML references `../log/best_model.onnx`, `../cal_data` and output prefix
`lanenet256x512_nv12`; its runtime/train inputs are **featuremap NCHW**. The helper
writes a separate config with absolute caller paths and prefix `lanenet256x512`.
Prepare-only checks the ONNX file and calibration identities; inspect graph and
output semantics during model validation.

<a id="calibration"></a>
## Explicit calibration preparation

Prepare calibration images from the
[TuSimple](https://github.com/TuSimple/tusimple-benchmark/issues/3) dataset locally.
Record the dataset license, split, image IDs, selection seed and transform code
with the calibration manifest.

From repository root, using an existing image directory and a new output path:

```bash
python -m samples.vision.lanenet.conversion.prepare_calibration \
  --images /work/tusimple/images --count 100 --output /work/lanenet/calibration
```

This selects the first100 image paths in lexicographic order from recursive
JPG/JPEG/PNG/BMP discovery. Count must be positive and available; an unreadable
selected image is an error. It uses the **same image-to-tensor function as the
runtime**: BGR→RGB, area resize256×512, float32 `/255`, per-channel mean
`[.485,.456,.406]` and std`[.229,.224,.225]`, then NCHW.

`data/000000.npy` etc. contain `[1,3,256,512]` float32 tensors. `manifest.json`
records source paths/hashes, actual image shapes, tensor shapes/dtypes/hashes,
count and protocol. Confirm whether
your ONNX includes normalization; applying it twice changes the model contract.

<a id="compile"></a>
## Prepare config, then compile

First inspect the generated configuration without invoking OE:

```bash
python -m samples.vision.lanenet.conversion.compile \
  --onnx /work/lanenet/best_model.onnx \
  --calibration-manifest /work/lanenet/calibration/manifest.json \
  --output /work/lanenet/config-review --prepare-only
```

Then, in a prepared compatible OE environment, use a different new output path:

```bash
python -m samples.vision.lanenet.conversion.compile \
  --onnx /work/lanenet/best_model.onnx \
  --calibration-manifest /work/lanenet/calibration/manifest.json \
  --output /work/lanenet/compiled
```

The helper checks declared/actual calibration shapes, float32 values, normalized
range, digests, unique entries and an exact data-directory file set. It uses the YAML
march/quantization/compiler settings and invokes `hb_compile -c` on the
generated config. It captures stdout/stderr and rejects nonzero exit or a missing/
empty expected HBM even if the compiler returns zero. `--prepare-only` never
calls a compiler; use the compilation command to produce HBM.

<a id="validation"></a>
## Validation and reference performance

Inspect public tensor names/layouts/dtypes, compare raw embeddings
and exact binary labels on identical inputs, record all auxiliary outputs and
check S100 execution. For lane-instance accuracy, cluster the embeddings into
instances and define matching rules against dataset labels.

The published three-output comparison describes high cosine similarity. The HRT
reference performance uses 200 frames, 14.245 ms average model latency and
69.894 FPS; firmware, model digest and complete parameters are unspecified.
Record those conditions when measuring this sample on the board. See
[evaluation](../evaluator/README.md).

<a id="artifacts"></a>
## Artifacts and provenance

Preparation writes `config.yaml` and `report.json`. A real successful compile
additionally requires `artifacts/lanenet256x512.hbm`, records its observed digest,
and retains `stdout.log`/`stderr.log`. The report identifies ONNX/calibration/
template/config hashes and the exact command. Its artifact origin is
`caller-converted; not published asset authentication`.

Use the exact S100 contract identity with the runtime's external-copy mode.
Changed shapes, names or numeric semantics require a matching binding.
[Model preparation](../model/README.md) describes published artifact paths.

<a id="known-gaps"></a>
## Preparation requirements

Prepare the export code/revision, checkpoint digest, calibration manifest, OE
version and accuracy reference with your model. Check generated configurations
before compilation, then validate ONNX outputs and the compiled model on S100.
