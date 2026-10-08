English | [简体中文](README_cn.md)

# ResNet model conversion (ResNet18 / ResNet50 / ResNet152)

Conversion runs on an x86 Linux host in the RDK OpenExplore (OE)
environment; it is not a board operation. This directory covers the three
variants this sample ships, each with its own conversion workflow:

| Variant | In this directory | Conversion workflow |
| --- | --- | --- |
| ResNet18 (x5 + s) | `export_resnet18_onnx.py` | run the supplied ONNX exporter on the host, then take calibration data and YAML from the OE `13_resnet18` example |
| ResNet50 (s only) | — | converted with the OE `13_resnet50` recipe |
| ResNet152 (s only) | `resnet152_config.yaml`, `get_calibration_data.py`, `x86_inference.py` | published ONNX plus user-provided calibration images, run through the three files above inside OE |

<a id="source-model"></a>
## Source model

All three variants are TorchVision ResNet family ImageNet-1k
classifiers: [ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html),
[ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html),
[ResNet152](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html).

- ResNet18: exported locally with fixed NCHW input `[1,3,224,224]`,
  output name `output`, ImageNet-1k classes. The exporter uses
  the TorchScript ONNX exporter (`dynamo=False`) for graph stability.
  `--weights IMAGENET1K_V1` selects the official pretrained weights;
  `--weights none` produces a random-weight graph for offline structure
  checks only.
- ResNet50: converted with the OE `13_resnet50` classification example —
  start there for the ONNX, calibration and compile steps.
- ResNet152: a published ONNX exists — inside the OE container (or any
  x86 host with the link reachable):

```bash
# cwd: this conversion directory — output: ./resnet152.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet152.onnx
```

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── export_resnet18_onnx.py  # Python script
├── get_calibration_data.py  # Python script
├── resnet152_config.yaml  # Configuration
└── x86_inference.py  # Python script
```

<a id="toolchain-targets"></a>
## Toolchain and targets

Use the OE Docker/toolchain release matching the target board and record
its image tag, OE version, and host date. Authoritative references:
[RDK S toolchain overview](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview),
[D-Robotics toolchain download](https://toolchain.d-robotics.cc/).
A generic load-and-mount sequence:

```bash
# inputs: OE image archive — run the rest inside the container at /workspace
export OE_IMAGE_TAR=/absolute/path/to/oe-image.tar
test -s "$OE_IMAGE_TAR"
docker load -i "$OE_IMAGE_TAR"
docker images
: "${OE_IMAGE:?Set OE_IMAGE to the loaded OE image:tag}"
export REPOSITORY_ROOT="${REPOSITORY_ROOT:-$PWD}"
docker run --rm -it --network host --shm-size=15g \
  -v "$REPOSITORY_ROOT":/workspace --workdir /workspace \
  "$OE_IMAGE" /bin/bash
```

Targets: X5 compiles with `hb_mapper` using march `bayes-e` (confirm
against the selected OE release); S100 uses `hb_compile` with `nash-e`,
S600 with `nash-p`. OE classification examples used by these workflows:
`13_resnet18` and `13_resnet50` under
`samples/ai_toolchain/horizon_model_convert_sample/03_classification/`.

<a id="export"></a>
## Export

ResNet18 — exporter prerequisites: PyTorch, TorchVision, ONNX (ONNX
Runtime optional for a host graph smoke test). Inside the OE container
at `/workspace`:

```bash
# input: TorchVision weights (downloads when not cached)
# output: $WORK/resnet18.onnx — success: onnx.checker passes, exit 0
export WORK="${WORK:-/workspace/resnet18-conversion}"
mkdir -p "$WORK"
python3 samples/vision/resnet/conversion/export_resnet18_onnx.py \
  --output "$WORK/resnet18.onnx" \
  --opset 11 \
  --weights IMAGENET1K_V1
```

`--weights none` variant for an offline smoke test (random weights, no
accuracy meaning). `--no-check` skips `onnx.checker` only when validation
is performed by another recorded tool. A host export check with Torch
2.7.1, TorchVision 0.22.1, ONNX 1.19 and ONNX Runtime 1.23.2 produces a
`[1,1000]` output matching the same seeded PyTorch graph within `1e-4`.

ResNet50 — export runs in the OE `13_resnet50` example. ResNet152 — use
the published ONNX from [Source model](#source-model); there is no
exporter to run.

<a id="calibration"></a>
## Calibration

ResNet18 — this directory supplies the exporter; calibration data and
YAML come from the OE `13_resnet18` example. Copy the matching
`13_resnet18` files from the OE sample and follow its documented
preprocessing and calibration procedure; record the exact calibration
data. The ResNet152 configuration in this directory is a ResNet152
recipe — it is not a ResNet18 recipe and must not be reused as one.

ResNet152 — `get_calibration_data.py` (run inside the OE container, cwd:
this conversion directory) converts 100 ImageNet validation images to
float32 RGB calibration data. Two inputs are user-provided and must be
edited in the script before running; the script's default paths point into
an external OE example tree; set them for your local data:

```python
# get_calibration_data.py — user-edited inputs
src_image_dir = '../../../open_explorer/samples/ai_toolchain/horizon_model_convert_sample/01_common/calibration_data/imagenet/'  # EDIT: point at your ILSVRC2012_val_*.JPEG directory
output_calib_dir = './calibration_data_rgb/'  # matches resnet152_config.yaml cal_data_dir
```

```bash
# cwd: this conversion directory (inside the OE container)
# input: src_image_dir with >=100 ILSVRC2012_val_*.JPEG
# output: ./calibration_data_rgb/*.npy (float32, RGB, NCHW) — success: 100 files printed
python3 get_calibration_data.py
```

The relative paths chain within this cwd: the script writes
`./calibration_data_rgb/`, which is exactly `resnet152_config.yaml`'s
`cal_data_dir`; the YAML's `onnx_model` `./resnet152.onnx` is the file
the [Source model](#source-model) step downloads into this directory,
and its `working_dir`/`output_model_file_prefix` produce
`./model_output/resnet152_224x224_nv12.hbm` as stated under
[Compile](#compile).

The normalization constants are **not identical** between the script and
the YAML: both use the same mean
(`123.675 116.28 103.53`), but the script multiplies by a uniform
`0.017` while the YAML declares per-channel `scale_value: 0.01712475
0.017507 0.01742919`. When regenerating, keep the script and the YAML on
one scale set; the YAML's per-channel values are the set that matches the
published record's Mean/Scale row (see [Validation](#validation)).

ResNet50 — calibration runs in the OE `13_resnet50` example; nothing to
run in this directory.

<a id="compile"></a>
## Compile

ResNet18 — inside the OE container, with `$WORK/resnet18.onnx` and
`$OE_CONFIG` (the OE sample's YAML) prepared. X5 block:

```bash
# output: *.bin under $WORK — success: hb_mapper exits 0
export X5_MARCH="${X5_MARCH:-bayes-e}"
hb_mapper --version
hb_mapper checker --model-type onnx --march "$X5_MARCH" --model "$WORK/resnet18.onnx"
hb_mapper makertbin --model-type onnx --config "$OE_CONFIG"
find "$WORK" -type f -name '*.bin' -print
```

S100/S600 block (the YAML must contain the matching Nash target — `nash-e`
for S100, `nash-p` for S600):

```bash
# output: *.hbm under $WORK — success: hb_compile exits 0
hb_compile --help
hb_compile --config "$OE_CONFIG"
find "$WORK" -type f -name '*.hbm' -print
```

ResNet152 — with `./resnet152.onnx` and `./calibration_data_rgb/` in
place (cwd: this conversion directory, inside the OE container):

```bash
# inputs: ./resnet152.onnx, ./calibration_data_rgb (see Calibration)
# output: ./model_output/resnet152_224x224_nv12.hbm — success: hb_compile exits 0
hb_compile --config resnet152_config.yaml
```

`resnet152_config.yaml` defaults to `march: nash-e` (S100). For S600,
change `march` to `nash-p` and keep every other field. The expected
output prefix is `resnet152_224x224_nv12` (matches the manifest
filename `resnet152_224x224_nv12.hbm`).

ResNet50 — compile via the OE `13_resnet50` example; this directory
ships no ResNet50 config. Run only the block for the artifact being
regenerated. If
the OE sample uses a different command spelling, run that exact
command and record it with the result.

For an ONNX path stored in `ONNX`, use the same X5 checker and compiler:

```bash
export ONNX="$WORK/resnet18.onnx"
export X5_MARCH="${X5_MARCH:-bayes-e}"
hb_mapper --version
hb_mapper checker --model-type onnx --march "$X5_MARCH" --model "$ONNX"
hb_mapper makertbin --model-type onnx --config "$OE_CONFIG"
find "$WORK" -type f -name '*.bin' -print
```

<a id="validation"></a>
## Validation

Use the OE sample's `hb_perf` and `hrt_model_exec` invocations with the
generated artifact and record their complete output. For ResNet152,
`x86_inference.py` (run inside the OE container) compares ONNX / HBIR
(`.bc`) / HBM outputs on x86 with the same padded-crop preprocessing.

Then confirm on the matching board with the runtime: X5
metadata exposes one packed NV12 input and an F32 `[1,1000,1,1]` output
named `prob`; S100/S600 expose Y `[1,224,224,1]`, UV `[1,112,112,2]`,
and an F32 `[1,1000]` output; the same image, resize type, label file,
and Top-K produce the expected class IDs and score ordering.

The published conversion record for ResNet152:

| Item | Value (published record) |
| --- | --- |
| Runtime input / train input | NV12 / RGB (NCHW) |
| Mean / Scale | `123.675 116.28 103.53` / `0.01712475 0.017507 0.01742919` |
| March | `nash-e` |
| Calibration similarity | `0.994397` |
| Quantization similarity | `0.992285` |
| Toolchain FPS / latency | `449.03` / `2.23 ms` |

Regenerating any artifact requires the OE compilation and the board checks
under [Validation](#validation).

<a id="artifacts"></a>
## Artifacts

| Stage | Artifact to retain |
| --- | --- |
| export | `resnet18.onnx`, exporter arguments, Torch/TorchVision/ONNX versions; ResNet152: downloaded `resnet152.onnx` and its origin URL |
| calibration | ResNet18: the OE sample's calibration directory/list and preprocessing record; ResNet152: `calibration_data_rgb/` and the exact source image list |
| checker | checker command, target config, log |
| compile | target config, OE version, generated output directory |
| deployment | X5 `resnet18_224x224_nv12.bin` or S `resnet{18,50,152}_224x224_nv12.hbm` (manifest filenames) |
| validation | `hb_perf`/`hrt_model_exec`/`x86_inference.py` output, raw output, latency, accuracy input |

The published filenames are the manifest rows used by this sample. The
exporter names its ONNX output `output`; the released X5 runtime metadata
names its output `prob` with shape `[1,1000,1,1]`, S artifacts expose
`output` with `[1,1000]` — these names and shapes are part of the target
artifact contract.

Published S-series artifacts can also be fetched directly from the model
server (S100 and S600 share the filename; only the archive sub-directory
differs — the manifest-driven [model downloader](../model/README.md#preparation)
remains the preparation path):

```bash
# ResNet18
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet18_224x224_nv12.hbm
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ResNet/resnet18_224x224_nv12.hbm
# ResNet50
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet50_224x224_nv12.hbm
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ResNet/resnet50_224x224_nv12.hbm
# ResNet152
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet152_224x224_nv12.hbm
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ResNet/resnet152_224x224_nv12.hbm
```

<a id="known-gaps"></a>
## Additional preparation

- ResNet18: obtain the calibration set, YAML and preprocessing record
  from the OE `13_resnet18` example.
- ResNet50: obtain the ONNX, calibration data and YAML from the OE
  `13_resnet50` example and run the conversion there.
- ResNet152: calibration images are user-provided — edit `src_image_dir`
  in `get_calibration_data.py` to point at your ILSVRC2012 validation
  directory; the script's uniform `0.017` scale differs from the YAML's
  per-channel `scale_value` — keep the script and the YAML on one
  consistent scale set when regenerating.
- The exact weights and OE configuration used for each published
  artifact are unrecorded; a regenerated file with the same basename is
  not equivalent until target, input metadata, output shape/dtype, and
  numerical results are compared.

<a id="self-trained"></a>
## Self-trained TorchVision ResNet18 checkpoints

The default exporter targets official `IMAGENET1K_V1` weights. A
self-trained `state_dict` exports through the same fixed contract:

```bash
# conversion environment (PyTorch + TorchVision installed); cwd: this directory
python3 export_resnet18_onnx.py \
  --checkpoint /path/to/my_resnet18.pth \
  --num-classes 4 \
  --output my_resnet18_4class.onnx
```

Rules and guarantees:

- CLI contract: `--checkpoint` and `--weights` are mutually exclusive
  (argparse rejects combining them); `--num-classes` is required with
  `--checkpoint` and rejected without it; when neither weight option is
  given, the official `IMAGENET1K_V1` default (1000-class output)
  applies unchanged.
- Architecture: TorchVision `resnet18` only. The classifier head is
  rebuilt as `nn.Linear(512, num_classes)` and the checkpoint must load
  with `strict=True`. Use a ResNet18 checkpoint whose classifier head
  has the requested class count.
- `--num-classes` (>= 2) sets the ONNX `output` width; the exporter
  prints the torch/torchvision versions of the exporting environment —
  record them with your training provenance.
- Fixed graph contract, identical to the official export: one NCHW
  input `data` `[1, 3, 224, 224]` float32, one output `output`
  `[1, num_classes]`, static shapes, TorchScript exporter
  (`dynamo=False`).
- Preprocessing: training must use a transform compatible with the OE
  conversion configuration you compile with — the same one the official
  artifact uses (letterbox/direct resize per `--resize-type`, the
  normalization baked into the OE config). The exporter does not alter
  preprocessing.
- Compile the resulting ONNX with the existing OE configurations
  documented under [Toolchain and targets](#toolchain-targets) (X5
  `hb_mapper` → `.bin`, S100/S600 `hb_compile` → `.hbm`); no published
  artifact recipe is implied for self-trained outputs.
- Runtime contract: load the compiled artifact with
  `classify.ResNetClassifier(model_path, target=target,
  input_size=(224, 224), class_count=<num-classes>)` and
  optional labels (see the [custom models
  section](../runtime/python/README.md#custom-model)); the binding
  validates the actual output width against `class_count`.
