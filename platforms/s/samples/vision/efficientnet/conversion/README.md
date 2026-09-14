English | [简体中文](./README_cn.md)

# EfficientNet-Lite Conversion

This directory provides the EfficientNet-Lite conversion notes and scripts used by this sample.

## Files

| File | Description |
| --- | --- |
| `efficientnet_lite*_config.yaml` | OE conversion configurations for Lite0 to Lite4. |
| `get_efficientnet_lite*_onnx.py` | ONNX export scripts for each EfficientNet-Lite variant. |
| `timm2onnx_local.py` | Local timm checkpoint to ONNX conversion helper used by this sample. |
| `get_calibration_data.py` | Calibration image preprocessing script used by this sample. |
| `x86_inference.py` | x86 reference inference script for ONNX/HBIR/HBM inspection. |

## Source Model

- Paper: `https://arxiv.org/abs/1905.11946`
- Source repository: `https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet`
- EfficientNet-Lite weights are exported through timm models such as `timm/tf_efficientnet_lite0.in1k`.

Install export dependencies when regenerating ONNX models:

```bash
pip install timm onnx
```

Run the matching export script, for example:

```bash
python3 get_efficientnet_lite0_onnx.py
```

## Calibration Data

These models use 100 ImageNet validation images for calibration. Generate RGB calibration data with:

```bash
python3 get_calibration_data.py
```

The YAML files expect calibration data in:

```text
./calibration_data_rgb
```

## Compile

Compile the required variant with the matching YAML:

```bash
hb_compile --config efficientnet_lite0_config.yaml
```

The runtime sample downloads published HBM artifacts, so rebuilding is only required when changing the source model or conversion settings.

## Model Configurations

| Variant | ONNX name | HBM output prefix | Input |
| --- | --- | --- | --- |
| Lite0 | `tf_efficientnet_lite0.onnx` | `efficientnet_lite0_224x224_nv12` | 224x224 |
| Lite1 | `tf_efficientnet_lite1.onnx` | `efficientnet_lite1_240x240_nv12` | 240x240 |
| Lite2 | `tf_efficientnet_lite2.onnx` | `efficientnet_lite2_260x260_nv12` | 260x260 |
| Lite3 | `tf_efficientnet_lite3.onnx` | `efficientnet_lite3_300x300_nv12` | 300x300 |
| Lite4 | `tf_efficientnet_lite4.onnx` | `efficientnet_lite4_380x380_nv12` | 380x380 |

All variants use NV12 runtime input, RGB training input, NCHW training layout, mean `127 127 127`, scale `0.007843 0.007843 0.007843`, and `nash-e` march.

> **Note on S600 builds**: The S600 publish under ``rdk_s600/EfficientNet/``
> is produced from the same source ONNX with the same quantization
> configuration — only ``march`` is changed to ``nash-p``.

## OE Toolchain

Run model conversion on an x86 Linux host with the RDK S100 OpenExplore environment. Model conversion is not intended to run on the board.

- OE resource entry point (Docker + OE development package): <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain online manual: <https://toolchain.d-robotics.cc/>

Download the OpenExplore CPU Docker image for the target RDK platform from
the OE resource entry point, then load the actual image file:

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

> To compile for S600, change ``march`` from ``nash-e`` to ``nash-p`` in the
> YAML config before running ``hb_compile``, or pass ``--march nash-p`` on
> the command line.
