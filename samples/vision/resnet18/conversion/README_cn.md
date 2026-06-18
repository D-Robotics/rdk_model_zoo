[English](./README.md) | 简体中文

# ResNet18 模型转换

当前 ResNet18 sample 没有独立 YAML 或 ONNX 导出脚本。转换说明记录
了源模型，并指向 OE SDK 的分类转换示例。

## 源模型

已发布 HBM 模型由 TorchVision ResNet18 ONNX 模型转换而来：

- TorchVision 模型页面：<https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html>
- PyTorch 实现：<https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>

## 原始转换参考

原始 README 说明，量化和转换步骤可参考 OE SDK 示例：

```text
samples/ai_toolchain/horizon_model_convert_sample/03_classification/13_resnet18
```

如需重新生成 HBM，以 OE SDK 示例作为权威转换流程。

## 运行模型

本 sample 使用的部署模型文件：

```text
resnet18_224x224_nv12.hbm
```

模型信息：

| 项目 | 值 |
| --- | --- |
| 运行时输入 | NV12 |
| 输入尺寸 | 224x224 |
| 目标 march | `nash-e`（RDK S100）/ `nash-p`（RDK S600）|
| 运行模型 | `../model/s100/resnet18_224x224_nv12.hbm`（S100）<br/>`../model/s600/resnet18_224x224_nv12.hbm`（S600）|

## 下载

原始下载 URL 保留在 `../model/download_model.sh`：

```bash
# RDK S100
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet18_224x224_nv12.hbm
# RDK S600
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ResNet/resnet18_224x224_nv12.hbm
```

## 模型说明

本 sample 使用公开 RDK ResNet18 HBM 模型，S100 和 S600 模型文件名相同，仅
archive 子目录不同。如需重新生成模型，请参考上面的 OE SDK 转换说明。

## OE 工具链

模型转换请在 x86 Linux 主机的 RDK OpenExplore 环境中完成，不建议在板端执行转换（S100/S600 共用同一套 OE 工具链）。

- OE Docker 下载文档：<https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE 工具链下载：<https://toolchain.d-robotics.cc/>

请从 OE Docker 下载文档获取适配目标 SoC（S100/S100P/S600）的 OpenExplore CPU Docker 镜像，并按实际文件名加载：

```bash
sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
sudo docker images
```

启动容器时建议挂载当前仓库并增大共享内存：

```bash
sudo docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  <docker-image-name> /bin/bash
```
[English](./README.md) | 简体中文

# ResNet18 模型转换

当前 ResNet18 sample 没有独立 YAML 或 ONNX 导出脚本。转换说明记录
了源模型，并指向 OE SDK 的分类转换示例。

## 源模型

已发布 HBM 模型由 TorchVision ResNet18 ONNX 模型转换而来：

- TorchVision 模型页面：<https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html>
- PyTorch 实现：<https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>

## 原始转换参考

原始 README 说明，量化和转换步骤可参考 OE SDK 示例：

```text
samples/ai_toolchain/horizon_model_convert_sample/03_classification/13_resnet18
```

如需重新生成 HBM，以 OE SDK 示例作为权威转换流程。

## 运行模型

本 sample 使用的部署模型文件：

```text
resnet18_224x224_nv12.hbm
```

模型信息：

| 项目 | 值 |
| --- | --- |
| 运行时输入 | NV12 |
| 输入尺寸 | 224x224 |
| 目标 march | `nash-e`（RDK S100）/ `nash-p`（RDK S600）|
| 运行模型 | `../model/s100/resnet18_224x224_nv12.hbm`（S100）<br/>`../model/s600/resnet18_224x224_nv12.hbm`（S600）|

## 下载

原始下载 URL 保留在 `../model/download_model.sh`：

```bash
# RDK S100
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet18_224x224_nv12.hbm
# RDK S600
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ResNet/resnet18_224x224_nv12.hbm
```

## 模型说明

本 sample 使用公开 RDK ResNet18 HBM 模型，S100 和 S600 模型文件名相同，仅
archive 子目录不同。如需重新生成模型，请参考上面的 OE SDK 转换说明。
