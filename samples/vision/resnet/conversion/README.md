# ResNet model conversion (ResNet18 / ResNet50 / ResNet152)

Conversion runs on an x86 Linux host in the RDK OpenExplore (OE)
environment; it is not a board operation. This directory covers the three
variants this sample ships, with a different reproducible scope per
variant — stated below and never overstated:

| Variant | In this directory | Reproducible scope |
| --- | --- | --- |
| ResNet18 (x5 + s) | `export_resnet18_onnx.py` | ONNX export replayable on host; calibration/YAML owned by the OE `13_resnet18` example (declared gap) |
| ResNet50 (s only) | — | No source recipe existed (rdk_s @380e1a2 shipped README pointers only); the OE `13_resnet50` example is the authority |
| ResNet152 (s only) | `resnet152_config.yaml`, `get_calibration_data.py`, `x86_inference.py` | Full OE recipe kept verbatim from the source branch; replayable inside OE given a published ONNX and user-provided calibration images |

The ResNet152 files are byte-verbatim from
`rdk_s@380e1a2:samples/vision/resnet152/conversion/` (audited source of
this batch; SHA-256 pinned by `tests/test_conversion_layout.py`).
Adjusting them is a source-branch change, not a local edit.

<a id="source-model"></a>
## Source model

All three variants are TorchVision ResNet family ImageNet-1k
classifiers: [ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html),
[ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html),
[ResNet152](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html).

- ResNet18: exported locally with fixed NCHW input `[1,3,224,224]`,
  output name `output`, ImageNet-1k classes. The canonical exporter uses
  the TorchScript ONNX exporter (`dynamo=False`) for graph stability.
  `--weights IMAGENET1K_V1` selects the official pretrained weights;
  `--weights none` produces a random-weight graph for offline structure
  checks only.
- ResNet50: the audited source published the HBM artifact and pointers
  to the OE classification example; no ONNX URL, YAML, or export script
  was published. This directory does not fabricate one.
- ResNet152: a published ONNX exists — inside the OE container (or any
  x86 host with the link reachable):

```bash
# cwd: this conversion directory — output: ./resnet152.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet152.onnx
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
S600 with `nash-p`. OE classification examples referenced by the
audited sources: `13_resnet18` and `13_resnet50` under
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
is performed by another recorded tool. The export smoke test has been run
with Torch 2.7.1, TorchVision 0.22.1, ONNX 1.19, and ONNX Runtime 1.23.2:
`[1,1000]` output matching the same seeded PyTorch graph within `1e-4`.
That is an ONNX structure check, not a BPU compilation or accuracy result.

ResNet50 — no export step is published for the audited artifact; start
from the OE `13_resnet50` example. ResNet152 — use the published ONNX
from [Source model](#source-model); there is no exporter to run.

<a id="calibration"></a>
## Calibration

ResNet18 — not reproducible from this repository (known gap): the
audited sources do not publish the calibration image list, mean/scale
values, or a ResNet18 YAML. Copy the matching `13_resnet18` files from
the OE sample and follow its documented preprocessing and calibration
procedure; record the exact calibration data. The ResNet152
configuration in this directory is a ResNet152 recipe — it is not a
ResNet18 recipe and must not be reused as one.

ResNet152 — `get_calibration_data.py` (run inside the OE container, cwd:
this conversion directory) converts 100 ImageNet validation images to
float32 RGB calibration data. Two inputs are user-provided and must be
edited in the script before running (kept verbatim from the source
branch, so its default paths are the legacy tree's):

```python
# get_calibration_data.py — user-edited inputs
src_image_dir = '../../../open_explorer/samples/ai_toolchain/horizon_model_convert_sample/01_common/calibration_data/imagenet/'  # EDIT: point at your ILSVRC2012_val_*.JPEG directory
output_calib_dir = './calibration_data_rgb/'  # matches resnet152_config.yaml cal_data_dir
```

```bash
# cwd: this conversion directory (inside the OE container)
# input: src_image_dir with >=100 ILSVRC2012_val_*.JPEG
# output: ./calibration_data_rgb/*.npy (float32, RGB, NCHW) — success: 100 files printed
python3 samples/vision/resnet/conversion/get_calibration_data.py
```

The transformer chain (padded center crop 224, resize, HWC2CHW, ×255,
mean `123.675 116.28 103.53`, ×0.017) matches the YAML's
`data_mean_and_scale` entries.

ResNet50 — calibration is owned by the OE `13_resnet50` example; nothing
to run here.

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

ResNet50 — compile via the OE `13_resnet50` example; no config is
shipped here. Run only the block for the artifact being regenerated. If
the OE sample uses a different command spelling, preserve that exact
command and file in the evidence record.

<a id="validation"></a>
## Validation

Use the OE sample's `hb_perf` and `hrt_model_exec` invocations with the
generated artifact and record their complete output. For ResNet152,
`x86_inference.py` (run inside the OE container) compares ONNX / HBIR
(`.bc`) / HBM outputs on x86 with the same padded-crop preprocessing.

Then confirm on the matching board with the canonical runtime: X5
metadata exposes one packed NV12 input and an F32 `[1,1000,1,1]` output
named `prob`; S100/S600 expose Y `[1,224,224,1]`, UV `[1,112,112,2]`,
and an F32 `[1,1000]` output; the same image, resize type, label file,
and Top-K produce the expected class IDs and score ordering.

The source branch's published conversion record for ResNet152 (context
only — not re-measured in this repository):

| Item | Value (rdk_s @380e1a2 record) |
| --- | --- |
| Runtime input / train input | NV12 / RGB (NCHW) |
| Mean / Scale | `123.675 116.28 103.53` / `0.01712475 0.017507 0.01742919` |
| March | `nash-e` |
| Calibration similarity | `0.994397` |
| Quantization similarity | `0.992285` |
| Toolchain FPS / latency | `449.03` / `2.23 ms` |

Status: ResNet18 export smoke test done (host); real OE compilation and
board re-validation of regenerated artifacts (any variant):
**not-run** — the published artifacts were not rebuilt in this sample.

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

<a id="known-gaps"></a>
## Known gaps

- ResNet18: no checked-in calibration set, YAML, or complete replay
  script for the published artifacts; the OE `13_resnet18` sample owns
  those steps.
- ResNet50: the audited source published no ONNX URL, YAML, calibration
  data, or scripts — only README pointers to the OE `13_resnet50`
  example. Regenerating this artifact is only possible from that
  example; this directory records the pointer instead of inventing a
  recipe.
- ResNet152: calibration images are user-provided (the script's default
  source directory belongs to the legacy branch tree and does not exist
  here); the S600 (`nash-p`) build and the published FPS/latency record
  have not been re-executed in this repository.
- The exact weights and OE configuration used for each published
  artifact are unrecorded; a regenerated file with the same basename is
  not equivalent until target, input metadata, output shape/dtype, and
  numerical results are compared.
- Real OE compilation in this sample has not been executed
  (**not-run**); only the host ResNet18 export smoke test is recorded.
