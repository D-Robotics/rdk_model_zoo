[English](README.md) | [简体中文](README_cn.md)

# LaneNet conversion

This directory preserves the source YAML and adds explicit calibration/config
preparation around it. The source export scripts are missing. Host fixture tests
exercise preparation and a fake compiler; no real ONNX export, OE compilation or
board execution has been performed in this migration.

<a id="source-model"></a>
## Source model and prerequisites

The source checkpoint URL is
[best_model.pth](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Lanenet/best_model.pth).
Its checksum, model-code revision and architecture binding are not supplied.
The source names `test.py --img ... --model best_model.pth` as an export step,
but `test.py` is absent. Obtain the matching model/export code before attempting
to reproduce the published artifact; a checkpoint filename alone cannot identify
its ONNX graph or preprocessing boundary.

The source lists Python≥3.6, Torch≥1.2, torchvision≥0.4.0, NumPy≥1.7, OpenCV,
pandas and matplotlib as historical export dependencies. These broad old bounds
are not a verified environment lock. The new host preparation tools need Python,
NumPy, OpenCV and PyYAML; they do not import Torch or a board SDK.

<a id="toolchain-targets"></a>
## Toolchain and target

The retained `config.yaml` declares **nash-e/S100**, latency mode, O2 and
`set_all_nodes_int16`. The source does not pin an OE Docker/compiler version.
Its resource links are [OE environment](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)
and [toolchain manual](https://toolchain.d-robotics.cc/). Prepare the matching x86
Linux toolchain explicitly. The wrapper never installs it.

Only S100 has a published model. A new S100P/S600 compilation needs its own
verified graph, march/configuration and runtime binding; changing a filename or
using S100's artifact is not such a port. Internal int16 quantization does not
specify the public embedding/binary tensor dtypes.

<a id="export"></a>
## Export boundary and retained template

Supply your own nonempty ONNX after resolving the missing export prerequisites.
Source input is float32 RGB NCHW `[1,3,256,512]`, INTER_AREA stretch, `/255` then
ImageNet mean/std. Runtime consumes named float32 `instance_seg_logits` and int64
`binary_seg_pred`; their actual artifact metadata remains unobserved.
Source conversion prose mentions three outputs but identifies no third one.
Do not invent its name or discard an observed auxiliary tensor.

The original YAML is byte-preserved. It references `../log/best_model.onnx`,
`../cal_data` and a misleading `lanenet256x512_nv12` output prefix even though
both runtime/train input types are **featuremap NCHW**, not NV12. The wrapper
writes a separate config with absolute caller paths and `lanenet256x512` prefix;
it does not edit the retained source file. Prepare-only checks file/calibration
identity, not ONNX graph correctness or output semantics.

<a id="calibration"></a>
## Explicit calibration preparation

The source cites TuSimple and an absent `get_calibration_data.py`. It does not
provide the original split, image IDs, seed or transform code. Obtain permitted
images locally; do not present newly chosen images as the source calibration set.

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
count and protocol. No model or dataset is downloaded. This is a new explicit
preparation workflow, not recovery of the missing source script. Confirm whether
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

The wrapper checks declared/actual calibration shapes, float32 values, normalized
range, digests, unique entries and an exact data-directory file set. It preserves
source march/quantization/compiler settings and invokes `hb_compile -c` on the
generated config. It captures stdout/stderr and rejects nonzero exit or a missing/
empty expected HBM even if the compiler returns zero. `--prepare-only` never
calls a compiler and is not a successful conversion claim.

<a id="validation"></a>
## Validation and historical evidence

A real run must inspect public tensor names/layouts/dtypes, compare raw embeddings
and exact binary labels on identical inputs, record all auxiliary outputs and
check actual S100 execution. Display colors cannot establish lane-instance
accuracy: the source has no clustering or stable lane-ID decoding.

The source says all three outputs have high cosine similarity but its referenced
`test_data/readme_img/result.jpg` is missing, with no numeric table, dataset or
raw logs. No replacement figure or similarity value is invented. Historical
performance is 200 frames,14.245ms average,69.894FPS from a source HRT record;
firmware, model digest and complete parameters are unspecified. It is not a
benchmark of the new pipeline. See [evaluation](../evaluator/README.md).

<a id="artifacts"></a>
## Artifacts and provenance

Preparation writes `config.yaml` and `report.json`. A real successful compile
additionally requires `artifacts/lanenet256x512.hbm`, records its observed digest,
and retains `stdout.log`/`stderr.log`. The report identifies ONNX/calibration/
template/config hashes and the exact command. Its artifact origin is
`caller-converted; not published asset authentication`.

The runtime's external-copy mode still requires exact contract identity and
actual S100 gating. The publisher hash is unknown; do not label arbitrary custom
weights as the original release. Changed shapes, names or numeric semantics
require an explicit new binding. [Model preparation](../model/README.md) explains
published paths separately.

<a id="known-gaps"></a>
## Remaining gaps

Missing source export code/revision/checkpoint digest, original calibration set,
OE version and accuracy figure remain missing. The original README also linked
an absent Chinese conversion page; this bilingual guide supplies the current
workflow. Generated configs and fake-compiler tests do not prove actual OE
compatibility, ONNX correctness or board parity. Those validations remain not-run.
