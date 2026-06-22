English | [简体中文](./README_cn.md)

# ResNet50 Model Conversion

The original S100 ResNet50 sample does not provide a dedicated YAML file or
ONNX export script in the sample directory. It documents the source model and
points to the OE SDK classification conversion example.

## Source Model

The released HBM model is converted from the TorchVision ResNet50 ONNX model:

- TorchVision model page: <https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html>
- PyTorch implementation: <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>

## Original Conversion Reference

The original README states that quantization and conversion can follow the OE
SDK sample:

```text
samples/ai_toolchain/horizon_model_convert_sample/03_classification/13_resnet50
```

Use the OE SDK sample as the authoritative conversion workflow when regenerating
the HBM model.

## Runtime Model

```text
resnet50_224x224_nv12.hbm
```

| Item | Value |
| --- | --- |
| Runtime input | NV12 |
| Input size | 224x224 |
| Target march | `nash-e` (RDK S100) / `nash-p` (RDK S600) |
| Runtime model | `../model/s100/resnet50_224x224_nv12.hbm` (S100) <br/> `../model/s600/resnet50_224x224_nv12.hbm` (S600) |

## Download

The original download URL is preserved in `../model/download_model.sh`:

```bash
# RDK S100
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet50_224x224_nv12.hbm
# RDK S600
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ResNet/resnet50_224x224_nv12.hbm
```

## Artifact Note

This sample uses the public RDK ResNet50 HBM model (S100 and S600 share the same
file name, only the archive sub-directory differs). Use the OE SDK conversion
reference above when regenerating the model.

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
