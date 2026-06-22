English | [简体中文](./README_cn.md)

# MobileNetV4 Model Conversion

This directory keeps the conversion assets from the original S100 MobileNetV4
sample and adapts the paths to the standard sample layout.

## Files

| File | Description |
| --- | --- |
| `mobilenetv4_small_config.yaml` | S100 conversion configuration for `mobilenetv4_conv_small.onnx` |
| `mobilenetv4_medium_config.yaml` | S100 conversion configuration for `mobilenetv4_conv_medium.onnx` |
| `get_mobilenetv4_onnx.py` | Export MobileNetV4 small and medium ONNX models from timm |
| `timm2onnx_local.py` | Legacy local-weight ONNX export helper; edit model name and weights path before use |
| `get_calibration_data.py` | Generate float32 BGR calibration `.npy` files |
| `x86_medium_inference.py` | Original x86 reference inference script for conversion-side verification |

## Source Model

The original sample exports the ONNX models from timm:

- `mobilenetv4_conv_small`, input `1x3x224x224`
- `mobilenetv4_conv_medium`, input `1x3x256x256`

Install the exporter dependencies:

```bash
pip install timm onnx onnxsim
```

If the pretrained weights are downloaded from HuggingFace, log in first:

```bash
huggingface-cli login
```

Export ONNX models:

```bash
python3 get_mobilenetv4_onnx.py
```

The original script records the expected export messages:

```text
Processing mobilenetv4_conv_small...
input: (3, 224, 224)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model saved to mobilenetv4_conv_small.onnx
Total number of parameters in the model: 3761480

Processing mobilenetv4_conv_medium...
input: (3, 256, 256)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model saved to mobilenetv4_conv_medium.onnx
Total number of parameters in the model: 9681560
```

For local weights, edit `timm2onnx_local.py` before running it. The file is
kept from the original sample as a reference helper.

## Calibration Data

The model uses ImageNet calibration images. The original sample expects 100
images named `ILSVRC2012_val_*.JPEG`.

```bash
python3 get_calibration_data.py
```

The script writes float32 calibration data. Select the target size in the script:

- `calibration_data_bgr_224` for the small model
- `calibration_data_bgr_256` for the medium model

## Compile

Quick ONNX verification:

```bash
hb_compile --model mobilenetv4_conv_small.onnx --march nash-e
hb_compile --model mobilenetv4_conv_medium.onnx --march nash-e
```

Compile with YAML:

```bash
hb_compile --config mobilenetv4_small_config.yaml
hb_compile --config mobilenetv4_medium_config.yaml
```

Key settings:

| Item | Small | Medium |
| --- | --- | --- |
| Source model | `mobilenetv4_conv_small.onnx` | `mobilenetv4_conv_medium.onnx` |
| Runtime input | NV12 | NV12 |
| Training input | BGR / NCHW | BGR / NCHW |
| Calibration data | `calibration_data_bgr_224` | `calibration_data_bgr_256` |
| Output prefix | `mobilenetv4_small_224x224_nv12` | `mobilenetv4_medium_256x256_nv12` |
| March | `nash-e` | `nash-e` |

The original `mobilenetv4_medium_config.yaml` referenced a 224 output prefix
while the shipped model and README use `mobilenetv4_medium_256x256_nv12.hbm`.
This migrated YAML follows the shipped S100 model name and the ONNX export
script's 256x256 medium input.

## Original Quantization Record

```text
mobilenetv4_medium:
Calibrated Cosine: 0.999759
Quantized Cosine: 0.999863

mobilenetv4_small:
Calibrated Cosine: 0.999892
Quantized Cosine: 0.99988
```

## Original Toolchain Performance Record

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

## Artifact Note

This sample uses the public S100 HBM models. Use the conversion reference above
when regenerating the models.

## Conversion Reference

- ONNX export
- PTQ configuration generation

## OE Resources

Run model conversion on an x86 Linux host with the RDK S100 OpenExplore environment. Model conversion is not intended to run on the board.

- OE resource entry point (Docker + OE development package): <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain online manual: <https://toolchain.d-robotics.cc/>

Download the OpenExplore CPU Docker image for RDK S100/S100P from the OE
resource entry point, then load the actual image file:

```bash
sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
sudo docker images
```

Start the container with the repository mounted and enough shared memory for compilation:

```bash
sudo docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  <docker-image-name> /bin/bash
```
English | [简体中文](./README_cn.md)

# MobileNetV4 Model Conversion

This directory keeps the conversion assets from the original S100 MobileNetV4
sample and adapts the paths to the standard sample layout.

## Files

| File | Description |
| --- | --- |
| `mobilenetv4_small_config.yaml` | S100 conversion configuration for `mobilenetv4_conv_small.onnx` |
| `mobilenetv4_medium_config.yaml` | S100 conversion configuration for `mobilenetv4_conv_medium.onnx` |
| `get_mobilenetv4_onnx.py` | Export MobileNetV4 small and medium ONNX models from timm |
| `timm2onnx_local.py` | Legacy local-weight ONNX export helper; edit model name and weights path before use |
| `get_calibration_data.py` | Generate float32 BGR calibration `.npy` files |
| `x86_medium_inference.py` | Original x86 reference inference script for conversion-side verification |

## Source Model

The original sample exports the ONNX models from timm:

- `mobilenetv4_conv_small`, input `1x3x224x224`
- `mobilenetv4_conv_medium`, input `1x3x256x256`

Install the exporter dependencies:

```bash
pip install timm onnx onnxsim
```

If the pretrained weights are downloaded from HuggingFace, log in first:

```bash
huggingface-cli login
```

Export ONNX models:

```bash
python3 get_mobilenetv4_onnx.py
```

The original script records the expected export messages:

```text
Processing mobilenetv4_conv_small...
input: (3, 224, 224)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model saved to mobilenetv4_conv_small.onnx
Total number of parameters in the model: 3761480

Processing mobilenetv4_conv_medium...
input: (3, 256, 256)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model saved to mobilenetv4_conv_medium.onnx
Total number of parameters in the model: 9681560
```

For local weights, edit `timm2onnx_local.py` before running it. The file is
kept from the original sample as a reference helper.

## Calibration Data

The model uses ImageNet calibration images. The original sample expects 100
images named `ILSVRC2012_val_*.JPEG`.

```bash
python3 get_calibration_data.py
```

The script writes float32 calibration data. Select the target size in the script:

- `calibration_data_bgr_224` for the small model
- `calibration_data_bgr_256` for the medium model

## Compile

Quick ONNX verification:

```bash
hb_compile --model mobilenetv4_conv_small.onnx --march nash-e
hb_compile --model mobilenetv4_conv_medium.onnx --march nash-e
```

Compile with YAML:

```bash
hb_compile --config mobilenetv4_small_config.yaml
hb_compile --config mobilenetv4_medium_config.yaml
```

Key settings:

| Item | Small | Medium |
| --- | --- | --- |
| Source model | `mobilenetv4_conv_small.onnx` | `mobilenetv4_conv_medium.onnx` |
| Runtime input | NV12 | NV12 |
| Training input | BGR / NCHW | BGR / NCHW |
| Calibration data | `calibration_data_bgr_224` | `calibration_data_bgr_256` |
| Output prefix | `mobilenetv4_small_224x224_nv12` | `mobilenetv4_medium_256x256_nv12` |
| March | `nash-e` | `nash-e` |

The original `mobilenetv4_medium_config.yaml` referenced a 224 output prefix
while the shipped model and README use `mobilenetv4_medium_256x256_nv12.hbm`.
This migrated YAML follows the shipped S100 model name and the ONNX export
script's 256x256 medium input.

## Original Quantization Record

```text
mobilenetv4_medium:
Calibrated Cosine: 0.999759
Quantized Cosine: 0.999863

mobilenetv4_small:
Calibrated Cosine: 0.999892
Quantized Cosine: 0.99988
```

## Original Toolchain Performance Record

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

## Artifact Note

This sample uses the public S100 HBM models. Use the conversion reference above
when regenerating the models.
