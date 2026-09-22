# MobileNetV3 model conversion

Conversion runs on an x86 Linux host in the RDK OpenExplore (OE)
environment; it is not a board operation. This directory keeps the
conversion material the source branches shipped and records the gaps
honestly; it does not invent a configuration that could produce a
different artifact.

<a id="source-model"></a>
## Source model

timm `mobilenetv3_large_100` with pretrained weights (MobileNetV3-Large),
fixed by `get_mobilenetv3_onnx.py`; NCHW input `[1,3,224,224]` on
both platforms.

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
`onnxsim`), cwd `samples/vision/mobilenetv3/conversion`:

```bash
# input: timm pretrained weights (downloads when not cached)
# output: the .onnx files below — success: onnx simplification check passes
python3 get_mobilenetv3_onnx.py    # -> ./mobilenetv3_large_100.onnx
```

The exporter uses onnx-simplifier and reports the parameter count. This
repository has not re-run it during the migration; treat the command as the
source branch's recorded recipe, not a verified result.
<a id="calibration"></a>
## Calibration

`get_calibration_data.py` is kept verbatim from the source branch. Its
hardcoded facts: it reads `ILSVRC2012_val_*.JPEG` from a legacy-tree
source directory
(`../../../open_explorer/samples/ai_toolchain/horizon_model_convert_sample/01_common/calibration_data/imagenet/`,
which does not exist in this repository — edit it to your own ImageNet
validation directory), and its transformer chain is fixed to BGR
(padded center crop 224, resize, HWC→CHW, `RGB2BGRTransformer`, ×255,
mean `103.94 116.78 123.68`, ×0.017), writing `./calibration_data_bgr/`.
The calibration images are not shipped; a regeneration must record the
exact image list used.

What each YAML consumes versus what the script produces:

| Config (target) | `cal_data_dir` | Layout/mean the YAML declares | Produced by the script as kept? |
| --- | --- | --- | --- |
| `mobilenetv3_s_config.yaml` (s100; s600 changes march only) | `./calibration_data_bgr` | BGR, mean `103.53 116.28 123.675` | **Yes.** Output dir and BGR chain match the recipe after the source-dir edit. The script's mean constants (`103.94 116.78 123.68`) differ slightly from the YAML's `103.53 116.28 123.675`; both files are source-verbatim — the discrepancy is disclosed here, not silently fixed. |
| `MobileNetV3_config.yaml` (x5) | `./calibration_data_rgb_f32` | RGB, mean `123.675 116.28 103.53` | **No — missing prerequisite.** The script has no RGB output mode, and no RGB calibration recipe for the X5 artifact was published. Renaming `calibration_data_bgr` to `calibration_data_rgb_f32` would feed BGR arrays with the wrong mean order to an RGB config — a rename is not a fix. The X5 calibration step is therefore not reproducible from this directory as kept. |

PTQ calibration has not been re-run in this repository.
<a id="compile"></a>
## Compile

Reference configurations kept in this directory:

| Config | Target | Inputs the config names | Command (inside the OE container) |
| --- | --- | --- | --- |
| `MobileNetV3_config.yaml` | x5 | `./mobilenetv3_large_100.onnx` (matches the exporter output), `./calibration_data_rgb_f32` (**missing** — see [Calibration](#calibration)) | `hb_mapper makertbin --config MobileNetV3_config.yaml` |
| `mobilenetv3_s_config.yaml` | s100 (s600: change march to `nash-p`) | `./mobilenetv3_large_100.onnx` (matches), `./calibration_data_bgr` (script-produced after the source-dir edit) | `hb_compile --config mobilenetv3_s_config.yaml` |

The march values are read from the YAML files themselves (`bayes-e` X5,
`nash-e` S100). Compilation has not been re-run in this repository; a
regenerated artifact is not equivalent to the published one until target,
input metadata, output shape/dtype, and numerical results are compared.
Rename note: the S source file was `mobilenetv3_config.yaml`;
on case-insensitive filesystems it collides with the X5 file
`MobileNetV3_config.yaml`, so the S copy here is
`mobilenetv3_s_config.yaml` — filename only, content verbatim.

<a id="validation"></a>
## Validation

For a regenerated artifact, run `hb_perf` and `hrt_model_exec` per the OE
manual and keep the complete output; then confirm on the matching board
with the canonical runtime that the contract holds: X5 exposes one packed
NV12 input and an F32 `[1,1000,1,1]` output; S100/S600 expose Y
`[1,224,224,1]`, UV `[1,112,112,2]`, and an F32 `[1,1000]` output; the
output semantics are raw logits (softmax applied by the runtime task). Validation status for regenerated
artifacts in this repository: **not-run**.

<a id="artifacts"></a>
## Kept material

- `get_mobilenetv3_onnx.py`
- `timm2onnx_local.py`
- `get_calibration_data.py`
- `MobileNetV3_config.yaml`
- `mobilenetv3_s_config.yaml`

<a id="known-gaps"></a>
## Known gaps

- The S-side YAML is renamed (`mobilenetv3_s_config.yaml`) for
case-insensitive filesystems; content is verbatim.
- **X5 calibration has no recipe**: `MobileNetV3_config.yaml` consumes
RGB `./calibration_data_rgb_f32`, which no kept script produces (the
calibrator is BGR-only) and which the source branch never published.
The X5 chain stops at this missing prerequisite.
- The calibrator's hardcoded source directory belongs to the legacy
tree; only the source-directory line (and nothing else) needs editing
before a run.
- The calibrator's mean constants differ slightly from the S YAML's
`mean_value` (source-verbatim inconsistency, disclosed above).
- End-to-end regeneration (export → calibration → compile → board
validation) has not been executed in this repository.
