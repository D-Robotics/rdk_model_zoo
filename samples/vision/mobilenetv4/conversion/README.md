# MobileNetV4 model conversion

Conversion runs on an x86 Linux host in the RDK OpenExplore (OE)
environment; it is not a board operation. This directory keeps the
conversion material the source branches shipped and records the gaps
honestly; it does not invent a configuration that could produce a
different artifact.

<a id="source-model"></a>
## Source model

timm `mobilenetv4_conv_small` and `mobilenetv4_conv_medium` with
pretrained weights, fixed by `get_mobilenetv4_onnx.py`. The script
exports small at `[1,3,224,224]` and medium at `[1,3,256,256]` —
see [known gaps](#known-gaps) for the X5 medium geometry.

<a id="toolchain-targets"></a>
## Toolchain and targets

Use the OE Docker/toolchain release matching the target board and record
its image tag, OE version, and host date. Authoritative references:
[RDK S toolchain overview](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview),
[D-Robotics toolchain download](https://toolchain.d-robotics.cc/).
Targets: X5 compiles with `hb_mapper` using march `bayes-e`; S100 uses
`hb_compile` with `nash-e`, S600 with `nash-p`. Mount the repository at
`/workspace` in the container with enough shared memory
(`--shm-size=15g`).

<a id="export"></a>
## Export

Inside the OE container (or any host with `torch`, `timm`, `onnx`, and
`onnxsim`), cwd `samples/vision/mobilenetv4/conversion`:

```bash
# input: timm pretrained weights (downloads when not cached)
# output: the .onnx files below — success: onnx simplification check passes
python3 get_mobilenetv4_onnx.py    # -> mobilenetv4_conv_small.onnx + mobilenetv4_conv_medium.onnx
```

The exporter uses onnx-simplifier and reports the parameter count. This
repository has not re-run it during the migration; treat the command as the
source branch's recorded recipe, not a verified result.
<a id="calibration"></a>
## Calibration

`get_calibration_data.py` is kept verbatim from the source branch. Its
hardcoded facts: it reads `ILSVRC2012_val_*.JPEG` from the same
legacy-tree source directory as the V3 calibrator
(`../../../open_explorer/samples/ai_toolchain/horizon_model_convert_sample/01_common/calibration_data/imagenet/`,
which does not exist in this repository — edit it to your own ImageNet
validation directory), and its transformer chain is BGR (padded center
crop, resize, HWC→CHW, `RGB2BGRTransformer`, ×255, mean
`103.94 116.78 123.68`, ×0.017) parameterized by an image size that the
script selects through two commented switch lines. The calibration
images are not shipped; a regeneration must record the exact image list.

What each YAML consumes versus what the script produces:

| Config (target) | `cal_data_dir` | Layout/size the YAML declares | How to produce it with the script as kept |
| --- | --- | --- | --- |
| `mobilenetv4_small_config.yaml` (s100; s600 changes march only) | `./calibration_data_bgr_224` | BGR, 224 | Defaults already match: `output_calib_dir = './calibration_data_bgr_224/'` and `data_transformer(224)` — edit only the source dir. |
| `mobilenetv4_medium_config.yaml` (s100; s600 changes march only) | `./calibration_data_bgr_256` | BGR, **256** | Switch the two commented lines marked by the script's own comments to `output_calib_dir = './calibration_data_bgr_256/'` and `active_transformers = data_transformer(256)` (the 224 pair is active by default), plus the source-dir edit. Both lines must switch together — a 256 dir fed with 224 data (or the reverse) is wrong. |
| `MobileNetV4_small.yaml` (x5) | `./calibration_data_rgb_f32` | RGB, 224 | **Missing prerequisite** — the script has no RGB output mode; renaming the BGR dir is not an RGB recipe. |
| `MobileNetV4_medium.yaml` (x5) | `./calibration_data_rgb_f32` | RGB, 224 | **Missing prerequisite** — same RGB gap. |

The script's mean constants (`103.94 116.78 123.68`) differ slightly
from the S-side YAMLs' `103.53 116.28 123.675`; both files are
source-verbatim — disclosed here, not silently fixed. PTQ calibration
has not been re-run in this repository.
<a id="compile"></a>
## Compile

Reference configurations kept in this directory:

| Config | Target | Inputs the config names | Command (inside the OE container) |
| --- | --- | --- | --- |
| `MobileNetV4_small.yaml` | x5 | `./mobilenetv4_conv_small.onnx` (matches the exporter output), `./calibration_data_rgb_f32` (**missing** — see [Calibration](#calibration)) | `hb_mapper makertbin --config MobileNetV4_small.yaml` |
| `MobileNetV4_medium.yaml` | x5 | `./mobilenetv4_conv_medium_deploy.onnx` (**no kept script produces this file**), `./calibration_data_rgb_f32` (**missing**) | `hb_mapper makertbin --config MobileNetV4_medium.yaml` |
| `mobilenetv4_small_config.yaml` | s100 (s600: march `nash-p`) | `./mobilenetv4_conv_small.onnx` (matches), `./calibration_data_bgr_224` (script-produced) | `hb_compile --config mobilenetv4_small_config.yaml` |
| `mobilenetv4_medium_config.yaml` | s100 (s600: march `nash-p`) | `./mobilenetv4_conv_medium.onnx` (matches the exporter's 256 export), `./calibration_data_bgr_256` (script-produced after the two-line switch) | `hb_compile --config mobilenetv4_medium_config.yaml` |

The **X5 medium config stacks three missing prerequisites**: it reads a
`mobilenetv4_conv_medium_deploy.onnx` that no kept script produces (the
exporter writes `mobilenetv4_conv_medium.onnx`, at 256x256 — see
[Source model](#source-model)), the published X5 medium artifact is
224x224, and its calibration dir must be RGB. Renaming the exported file
to the deploy name does not reconcile the geometry or the color order;
the X5 medium chain is not reproducible from this directory as kept and
no unconditional recipe is claimed for it.

S600 variants change only the march (`nash-p`) in the S-side YAML. The
march values above are read from the YAML files themselves (`bayes-e` X5,
`nash-e` S100). Compilation has not been re-run in this repository; a
regenerated artifact is not equivalent to the published one until target,
input metadata, output shape/dtype, and numerical results are compared.
Geometry note: the S-side medium config compiles the 256x256
input recorded in `mobilenetv4_medium_config.yaml`; the X5 medium
config builds the published 224x224 artifact. Both geometries are
real and the runtime contract table records them per target.

<a id="validation"></a>
## Validation

For a regenerated artifact, run `hb_perf` and `hrt_model_exec` per the OE
manual and keep the complete output; then confirm on the matching board
with the canonical runtime that the contract holds. Input shapes are
per target×variant — they mirror the runtime contract table and the
published artifact names:

| Target / variant | Input the metadata exposes | Output |
| --- | --- | --- |
| x5, small and medium | one packed NV12 input, 224x224 (`MobileNetV4_conv_{small,medium}_224x224_nv12.bin`) | F32 `[1,1000,1,1]` |
| s100/s600, small | Y `[1,224,224,1]`, UV `[1,112,112,2]` (`mobilenetv4_small_224x224_nv12.hbm`) | F32 `[1,1000]` |
| s100/s600, medium | **Y `[1,256,256,1]`, UV `[1,128,128,2]`** (`mobilenetv4_medium_256x256_nv12.hbm`) | F32 `[1,1000]` |

Accepting a correct S medium artifact with the 224 shapes above would be
a validation error — the medium S input is 256x256. The output semantics
are raw logits (softmax applied by the runtime task). Validation status
for regenerated artifacts in this repository: **not-run**.

<a id="artifacts"></a>
## Kept material

- `get_mobilenetv4_onnx.py`
- `timm2onnx_local.py`
- `get_calibration_data.py`
- `MobileNetV4_small.yaml`
- `MobileNetV4_medium.yaml`
- `mobilenetv4_small_config.yaml`
- `mobilenetv4_medium_config.yaml`
- `x86_medium_inference.py`

<a id="known-gaps"></a>
## Known gaps

- The X5-side export script generates the medium ONNX at 256x256 while
the published X5 medium artifact and its config are 224x224; the
source does not record how that reconciliation happened. Regenerating
the X5 medium artifact byte-comparably is therefore not proven.
- The X5 medium config additionally expects a `mobilenetv4_conv_medium_deploy.onnx`
input that no kept script produces — see [Compile](#compile).
- **X5 calibration (small and medium) has no recipe**: both X5 YAMLs
consume RGB `./calibration_data_rgb_f32`, which no kept script produces
(the calibrator is BGR-only) and which the source branch never published.
- The calibrator's hardcoded source directory belongs to the legacy
tree; a run needs the source-dir edit, and for the S medium recipe the
two commented 256-switch lines must be switched together.
- The calibrator's mean constants differ slightly from the S YAMLs'
`mean_value` (source-verbatim inconsistency, disclosed above).
- `x86_medium_inference.py` is a host ONNX reference used while auditing
the medium variant, not a deployment path.
- End-to-end regeneration has not been executed in this repository.
