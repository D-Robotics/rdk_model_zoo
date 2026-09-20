# ResNet18 model conversion

Conversion runs on an x86 Linux host in the RDK OpenExplore (OE)
environment; it is not a board operation. The audited X5 and S18 source
directories identify TorchVision ResNet18 and point to the OE
classification example, but neither ships the original YAML, calibration
set, or a complete replay script. This directory supplies the source
export step and records the OE-owned steps without inventing a
configuration that could produce a different artifact.

<a id="source-model"></a>
## Source model

TorchVision ResNet18 ([upstream](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)),
exported with the fixed NCHW input `[1,3,224,224]`, output name `output`,
and ImageNet-1k class count. The canonical exporter uses the TorchScript
ONNX exporter (`dynamo=False`) for graph stability. `--weights
IMAGENET1K_V1` selects the official pretrained weights (downloaded from
the TorchVision weight registry when not cached); `--weights none`
produces a random-weight graph that is only useful for an offline
structure check.

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
S600 with `nash-p`. The OE classification example referenced by the
audited sources is
`samples/ai_toolchain/horizon_model_convert_sample/03_classification/13_resnet18`.

<a id="export"></a>
## Export

Exporter prerequisites: PyTorch, TorchVision, ONNX (ONNX Runtime optional
for a host graph smoke test). Inside the OE container at `/workspace`:

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

<a id="calibration"></a>
## Calibration

Not reproducible from this repository (known gap): the audited sources do
not publish the calibration image list, mean/scale values, or a ResNet18
YAML. Copy the matching `13_resnet18` files from the OE sample and follow
its documented preprocessing and calibration procedure; record the exact
calibration data. The ResNet152 configuration under another legacy
directory is not a ResNet18 recipe. Do not present a regenerated model as
equivalent until the calibration inputs are recorded.

<a id="compile"></a>
## Compile

Inside the OE container, with `$WORK/resnet18.onnx` and `$OE_CONFIG` (the
OE sample's YAML) prepared. X5 block:

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

Run only the block for the artifact being regenerated. If the OE sample
uses a different command spelling, preserve that exact command and file in
the evidence record.

<a id="validation"></a>
## Validation

Use the OE sample's `hb_perf` and `hrt_model_exec` invocations with the
generated artifact and record their complete output. Then confirm on the
matching board with the canonical runtime: X5 metadata exposes one packed
NV12 input and an F32 `[1,1000,1,1]` output named `prob`; S100/S600 expose
Y `[1,224,224,1]`, UV `[1,112,112,2]`, and an F32 `[1,1000]` output; the
same image, resize type, label file, and Top-K produce the expected class
IDs and score ordering. Status: export smoke test done (host); real OE
compilation and board re-validation of a regenerated artifact:
**not-run** — the published artifacts were not rebuilt in this sample.

<a id="artifacts"></a>
## Artifacts

| Stage | Artifact to retain |
| --- | --- |
| export | `resnet18.onnx`, exporter arguments, Torch/TorchVision/ONNX versions |
| calibration | the OE sample's calibration directory/list and preprocessing record |
| checker | checker command, target config, log |
| compile | target config, OE version, generated output directory |
| deployment | X5 `resnet18_224x224_nv12.bin` or S `resnet18_224x224_nv12.hbm` (manifest filenames) |
| validation | `hb_perf`/`hrt_model_exec` output, raw output, latency, accuracy input |

The published filenames are the manifest rows used by this sample. The
exporter names its ONNX output `output`; the released X5 runtime metadata
names its output `prob` with shape `[1,1000,1,1]`, S artifacts expose
`output` with `[1,1000]` — these names and shapes are part of the target
artifact contract.

<a id="known-gaps"></a>
## Known gaps

- No checked-in calibration set, YAML, or complete replay script for the
  published artifacts; the OE `13_resnet18` sample owns those steps.
- The exact weights and OE configuration used for each published artifact
  are unrecorded; a regenerated file with the same basename is not
  equivalent until target, input metadata, output shape/dtype, and
  numerical results are compared.
- Real OE compilation of this exporter's ONNX has not been executed
  (**not-run**); only the host export smoke test is recorded.
