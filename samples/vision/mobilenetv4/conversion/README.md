# MobileNetV4 model conversion

Conversion runs on an x86 Linux host in the RDK OpenExplore (OE)
environment; it is not a board operation. This directory keeps the
conversion material the source deliveries shipped and records its gaps;
no configuration that could produce a different artifact is invented.

<a id="source-model"></a>
## Source model

timm `mobilenetv4_conv_small` and `mobilenetv4_conv_medium` with
pretrained weights, fixed by `get_mobilenetv4_onnx.py`. The script
exports small at `[1,3,224,224]` and medium at `[1,3,256,256]` —
see [additional preparation](#known-gaps) for the X5 medium geometry.

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

The exporter uses onnx-simplifier and reports the parameter count
(small 3,761,480 / medium 9,681,560).
<a id="calibration"></a>
## Calibration

The calibration helper reads `ILSVRC2012_val_*.JPEG` from the configured
`src_image_dir`. The source default is
`../../../open_explorer/samples/ai_toolchain/horizon_model_convert_sample/01_common/calibration_data/imagenet/`;
set `src_image_dir` to your local ImageNet validation directory before running it. Its BGR preprocessing applies padded center crop,
resize, HWC→CHW, `RGB2BGRTransformer`, ×255, mean `103.94 116.78 123.68`,
and ×0.017. Two commented switch lines select the image geometry. Select
and record the validation images used for calibration.

Prepare the inputs for each YAML:

| Config (target) | `cal_data_dir` | Layout and size | Preparation |
| --- | --- | --- | --- |
| `mobilenetv4_small_config.yaml` (s100; s600 changes march only) | `./calibration_data_bgr_224` | BGR, 224 | Keep the helper's 224 output directory and `data_transformer(224)` selection; set the source directory and align mean constants (`103.94 116.78 123.68`) with the S YAML values (`103.53 116.28 123.675`). |
| `mobilenetv4_medium_config.yaml` (s100; s600 changes march only) | `./calibration_data_bgr_256` | BGR, 256 | Switch both commented lines to `output_calib_dir = './calibration_data_bgr_256/'` and `active_transformers = data_transformer(256)`; set the source directory and align mean constants with the S YAML values. |
| `MobileNetV4_small.yaml` (x5) | `./calibration_data_rgb_f32` | RGB, 224 | Prepare float32 RGB calibration arrays at 224x224 using the X5 YAML channel order and normalization. |
| `MobileNetV4_medium.yaml` (x5) | `./calibration_data_rgb_f32` | RGB, 224 | Prepare float32 RGB calibration arrays at 224x224 using the X5 YAML channel order and normalization. |

<a id="compile"></a>
## Compile

Build configurations:

| Config | Target | Inputs the config names | Command (inside the OE container) |
| --- | --- | --- | --- |
| `MobileNetV4_small.yaml` | x5 | `./mobilenetv4_conv_small.onnx`, `./calibration_data_rgb_f32` (prepare as described in Calibration) | `hb_mapper makertbin --config MobileNetV4_small.yaml` |
| `MobileNetV4_medium.yaml` | x5 | `./mobilenetv4_conv_medium_deploy.onnx` (224x224 graph), `./calibration_data_rgb_f32` (prepare as described in Calibration) | `hb_mapper makertbin --config MobileNetV4_medium.yaml` |
| `mobilenetv4_small_config.yaml` | s100 (s600: march `nash-p`) | `./mobilenetv4_conv_small.onnx` (matches), `./calibration_data_bgr_224` (script-produced) | `hb_compile --config mobilenetv4_small_config.yaml` |
| `mobilenetv4_medium_config.yaml` | s100 (s600: march `nash-p`) | `./mobilenetv4_conv_medium.onnx` (matches the exporter's 256 export), `./calibration_data_bgr_256` (script-produced after the two-line switch) | `hb_compile --config mobilenetv4_medium_config.yaml` |

For the X5 medium build, provide `mobilenetv4_conv_medium_deploy.onnx` at
224x224 and RGB float32 calibration data in `./calibration_data_rgb_f32`.
The exporter above writes `mobilenetv4_conv_medium.onnx` at 256x256; export
or obtain a separate 224x224 graph for this YAML. Renaming the 256x256 file
does not change its input geometry.

S600 variants change only the march (`nash-p`) in the S-side YAML. The
march values above are read from the YAML files themselves (`bayes-e` X5,
`nash-e` S100). For each rebuild, compare the target, input metadata, output
shape/dtype, and numerical results before deployment.
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
are raw logits (softmax applied by the runtime task).

Published quantization record of the original S build (cosine similarity
after quantization):

```text
mobilenetv4_medium:
Calibrated Cosine: 0.999759
Quantized Cosine: 0.999863

mobilenetv4_small:
Calibrated Cosine: 0.999892
Quantized Cosine: 0.99988
```

Toolchain performance record of the original S build:

```text
mobilenetv4_medium:
FPS (1 core): 2468.07
latency: 0.41 ms (405.2 us)
BPU conv original OPs per run: 2,160,488,448

mobilenetv4_small:
FPS (1 core): 5698.18
latency: 0.18 ms (175.5 us)
BPU conv original OPs per run: 372,011,136
```

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
## Additional preparation

- The X5 medium YAML expects a 224x224 input graph named `mobilenetv4_conv_medium_deploy.onnx`; the retained medium export helper produces a 256x256 graph. Before compiling X5 medium, reconcile the ONNX geometry with the YAML and prepare calibration data using the same geometry.
- X5 small and medium calibration directories are RGB `./calibration_data_rgb_f32`; the retained calibrator produces BGR data. Convert channel order and apply the X5 YAML normalization before compiling.
- Set the calibration helper's source-image directory to the local dataset path. For S medium, switch both commented 256 geometry lines together. Align the helper mean constants with the selected S YAML's `mean_value`.
- `x86_medium_inference.py` runs the medium ONNX model on the host; use the compiled HBM with the board runtime for deployment.
