[English](./README.md) | 简体中文

# ResNet152 模型转换

本目录提供 ResNet152 模型的校准数据准备、转换配置和 HBM 编译说明。提供的 `resnet152_config.yaml` 默认 `march: nash-e`（RDK S100）；如需为 RDK S600 重新编译，将 `march` 改为 `nash-p` 即可。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `resnet152_config.yaml` | NV12 HBM 模型的 OE 转换配置。 |
| `get_calibration_data.py` | 转换流程中的校准图片预处理脚本。 |
| `x86_inference.py` | 原始 x86 参考推理脚本，用于 ONNX/HBIR/HBM 检查。 |

## 源模型

- 模型系列：TorchVision ResNet152。
- 模型参考：`https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html`
- 原始 ONNX 下载命令：

```bash
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet152.onnx
```

转换前请将 `resnet152.onnx` 放在本目录。

## 校准数据

本示例使用 100 张 ImageNet 验证集图片作为校准数据。使用以下命令生成 RGB 校准数据：

```bash
python3 get_calibration_data.py
```

YAML 中的校准数据路径为：

```text
./calibration_data_rgb
```

## 编译

```bash
hb_compile --config resnet152_config.yaml
```

预期输出前缀为：

```text
resnet152_224x224_nv12
```

运行示例默认下载已发布的 HBM 模型；只有修改源模型或转换配置时才需要重新编译。

## 原始转换记录

| 项目 | 数值 |
| --- | --- |
| 运行时输入类型 | NV12 |
| 训练输入类型 | RGB |
| 训练布局 | NCHW |
| Mean | `123.675 116.28 103.53` |
| Scale | `0.01712475 0.017507 0.01742919` |
| March | `nash-e` |
| 校准相似度 | `0.994397` |
| 量化相似度 | `0.992285` |
| 工具链 FPS | `449.03` |
| 工具链延迟 | `2.23 ms` |

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
