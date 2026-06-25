English | [简体中文](./README_cn.md)

# Model Conversion

The Model Zoo provides pre-compiled HBM models for MobileNetV1 on RDK S100
and RDK S600. Users who only need runtime inference can download the model
from `../model/`.

> The notes below describe the **S100** conversion path (`march: "nash-e"`).
> The S600 build is produced from the same source ONNX with the same
> quantization configuration; only `march` is changed to `nash-p` and the
> output is uploaded under `rdk_s600/MobileNet/` on the archive.

## Published Artifact

| File | Platform | Input | Runtime |
| --- | --- | --- | --- |
| `mobilenetv1_224x224_nv12.hbm` | S100 / S600 | 224x224 NV12 (Y + UV) | `hbm_runtime` |

## Regeneration Notes

MobileNetV1 uses the MobileNet-Caffe source model and is converted with the RDK
S100 OpenExplore toolchain. If the model needs to be regenerated, use the S100
OE package model conversion environment and keep the published runtime interface
unchanged: two NV12 inputs, Y plane and UV plane.

## Conversion Reference

- ONNX export
- PTQ configuration generation

## OE Resources

Run model conversion on an x86 Linux host with the RDK S100 OpenExplore
environment. Model conversion is not intended to run on the board.

- OE resource entry point (Docker + OE development package): <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain online manual: <https://toolchain.d-robotics.cc/>

Download the OpenExplore CPU Docker image for RDK S100/S100P from the OE
resource entry point, then load the actual image file:

```bash
sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
sudo docker images
```

Start the container with the repository mounted and enough shared memory for
compilation:

```bash
sudo docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  <docker-image-name> /bin/bash
```
