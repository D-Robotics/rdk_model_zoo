# MobileNetV3 model conversion

Conversion runs on an x86 Linux host in the RDK OpenExplore (OE)
environment; it is not a board operation. This directory keeps the
conversion material the source deliveries shipped and records its gaps;
no configuration that could produce a different artifact is invented.

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

The exporter uses onnx-simplifier and reports the parameter count
(5,470,832 parameters for `mobilenetv3_large_100`); the expected metadata
print is `mean (0.485, 0.456, 0.406)`, `std (0.229, 0.224, 0.225)`,
"Simplified model is valid.".
<a id="calibration"></a>
## Calibration

The calibration helper reads `ILSVRC2012_val_*.JPEG` from the configured
`src_image_dir`. The source default is
`../../../open_explorer/samples/ai_toolchain/horizon_model_convert_sample/01_common/calibration_data/imagenet/`;
set `src_image_dir` to your local ImageNet validation directory before running it. Preprocessing produces BGR data with padded center crop
224, resize, HWC→CHW, `RGB2BGRTransformer`, ×255, mean
`103.94 116.78 123.68`, and ×0.017, written to `./calibration_data_bgr/`.
Select and record the validation images used for the calibration set.

Prepare inputs for each YAML as follows:

| Config (target) | `cal_data_dir` | Layout and values | Preparation |
| --- | --- | --- | --- |
| `mobilenetv3_s_config.yaml` (s100; s600 changes march only) | `./calibration_data_bgr` | BGR, mean `103.53 116.28 123.675` | Set `src_image_dir` to the local validation directory and align the helper mean constants (`103.94 116.78 123.68`) with the YAML values before generating the BGR data. |
| `MobileNetV3_config.yaml` (x5) | `./calibration_data_rgb_f32` | RGB, mean `123.675 116.28 103.53` | Prepare float32 RGB calibration arrays at 224x224 using the X5 YAML's channel order and normalization; use a separate RGB preprocessing chain. |

<a id="compile"></a>
## Compile

Build configurations:

| Config | Target | Inputs the config names | Command (inside the OE container) |
| --- | --- | --- | --- |
| `MobileNetV3_config.yaml` | x5 | `./mobilenetv3_large_100.onnx`, `./calibration_data_rgb_f32` (prepare as described in Calibration) | `hb_mapper makertbin --config MobileNetV3_config.yaml` |
| `mobilenetv3_s_config.yaml` | s100 (s600: change march to `nash-p`) | `./mobilenetv3_large_100.onnx` (matches), `./calibration_data_bgr` (script-produced after the source-dir edit) | `hb_compile --config mobilenetv3_s_config.yaml` |

The march values are read from the YAML files themselves (`bayes-e` X5,
`nash-e` S100). For each rebuild, compare the target, input metadata, output
shape/dtype, and numerical results before deployment.
Use the exact target config filename: `mobilenetv3_s_config.yaml` for S100/S600
and `MobileNetV3_config.yaml` for X5. Keep the case distinction when selecting
the target YAML.

<a id="validation"></a>
## Validation

For a regenerated artifact, run `hb_perf` and `hrt_model_exec` per the OE
manual and keep the complete output; then confirm on the matching board
with the canonical runtime that the contract holds: X5 exposes one packed
NV12 input and an F32 `[1,1000,1,1]` output; S100/S600 expose Y
`[1,224,224,1]`, UV `[1,112,112,2]`, and an F32 `[1,1000]` output; the
output semantics are raw logits (softmax applied by the runtime task).

Published quantization record of the original S build (cosine similarity
after quantization):

```text
TensorName: output
Calibrated Cosine: 0.911233
Quantized Cosine: 0.909042
```

Toolchain performance reference of the original S build:

```text
FPS (1 core): 2616.81
latency: 0.38 ms (382.1 us)
BPU conv original OPs per run: 433,179,520
```

<a id="artifacts"></a>
## Kept material

- `get_mobilenetv3_onnx.py`
- `timm2onnx_local.py`
- `get_calibration_data.py`
- `MobileNetV3_config.yaml`
- `mobilenetv3_s_config.yaml`

<a id="known-gaps"></a>
## Additional preparation

- Before S-side export on a case-insensitive filesystem, use the YAML filename `mobilenetv3_s_config.yaml` shown above.
- For X5, prepare `./calibration_data_rgb_f32` in RGB order for `MobileNetV3_config.yaml`; the retained calibration helper produces BGR data, so convert channel order and apply the X5 YAML's RGB normalization rather than renaming the BGR directory.
- Set the calibration helper's source-image directory to the local ImageNet directory before running it. The helper's mean constants differ slightly from the S YAML's `mean_value`; keep the selected target's calibration values aligned with its config.
- Use the board runtime after compilation to check the emitted model contract.
