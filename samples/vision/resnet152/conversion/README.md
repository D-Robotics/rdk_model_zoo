English | [简体中文](./README_cn.md)

# ResNet152 Conversion

This directory preserves the conversion materials from the original RDK ResNet152 demo and places them in the standard sample layout. The provided `resnet152_config.yaml` targets `nash-e` (RDK S100); switch `march` to `nash-p` when recompiling for RDK S600.

## Files

| File | Description |
| --- | --- |
| `resnet152_config.yaml` | OE conversion configuration for the NV12 HBM model. |
| `get_calibration_data.py` | Calibration image preprocessing script used by this sample. |
| `x86_inference.py` | Original x86 reference inference script for ONNX/HBIR/HBM inspection. |

## Source Model

- Model family: TorchVision ResNet152.
- Model reference: `https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html`
- Original ONNX download command:

```bash
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet152.onnx
```

Place `resnet152.onnx` in this directory before running the conversion command.

## Calibration Data

This sample uses 100 ImageNet validation images for calibration. Generate RGB calibration data with:

```bash
python3 get_calibration_data.py
```

The YAML expects calibration data in:

```text
./calibration_data_rgb
```

## Compile

```bash
hb_compile --config resnet152_config.yaml
```

The expected output prefix is:

```text
resnet152_224x224_nv12
```

The runtime sample downloads the published HBM artifact, so rebuilding the model is only required when changing the source model or conversion settings.

## Original Conversion Records

| Item | Value |
| --- | --- |
| Runtime input type | NV12 |
| Train input type | RGB |
| Train layout | NCHW |
| Mean | `123.675 116.28 103.53` |
| Scale | `0.01712475 0.017507 0.01742919` |
| March | `nash-e` |
| Calibration similarity | `0.994397` |
| Quantized similarity | `0.992285` |
| Toolchain FPS | `449.03` |
| Toolchain latency | `2.23 ms` |

## OE Toolchain

Run model conversion on an x86 Linux host with the RDK OpenExplore environment (S100 and S600 share the same toolchain). Model conversion is not intended to run on the board.

- OE Docker documentation: <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain download: <https://toolchain.d-robotics.cc/>

Download the OpenExplore CPU Docker image for the target SoC (S100/S100P/S600) from the OE Docker documentation, then load the actual image file:

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

# ResNet152 Conversion

This directory preserves the conversion materials from the original RDK ResNet152 demo and places them in the standard sample layout. The provided `resnet152_config.yaml` targets `nash-e` (RDK S100); switch `march` to `nash-p` when recompiling for RDK S600.

## Files

| File | Description |
| --- | --- |
| `resnet152_config.yaml` | OE conversion configuration for the NV12 HBM model. |
| `get_calibration_data.py` | Calibration image preprocessing script used by this sample. |
| `x86_inference.py` | Original x86 reference inference script for ONNX/HBIR/HBM inspection. |

## Source Model

- Model family: TorchVision ResNet152.
- Model reference: `https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html`
- Original ONNX download command:

```bash
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet152.onnx
```

Place `resnet152.onnx` in this directory before running the conversion command.

## Calibration Data

This sample uses 100 ImageNet validation images for calibration. Generate RGB calibration data with:

```bash
python3 get_calibration_data.py
```

The YAML expects calibration data in:

```text
./calibration_data_rgb
```

## Compile

```bash
hb_compile --config resnet152_config.yaml
```

The expected output prefix is:

```text
resnet152_224x224_nv12
```

The runtime sample downloads the published HBM artifact, so rebuilding the model is only required when changing the source model or conversion settings.

## Original Conversion Records

| Item | Value |
| --- | --- |
| Runtime input type | NV12 |
| Train input type | RGB |
| Train layout | NCHW |
| Mean | `123.675 116.28 103.53` |
| Scale | `0.01712475 0.017507 0.01742919` |
| March | `nash-e` |
| Calibration similarity | `0.994397` |
| Quantized similarity | `0.992285` |
| Toolchain FPS | `449.03` |
| Toolchain latency | `2.23 ms` |
