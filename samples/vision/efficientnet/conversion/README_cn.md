[English](./README.md) | 简体中文

# EfficientNet-Lite 模型转换

本目录提供 EfficientNet-Lite 模型的 ONNX 导出、校准数据准备和 HBM 编译说明。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `efficientnet_lite*_config.yaml` | Lite0 到 Lite4 的 OE 转换配置。 |
| `get_efficientnet_lite*_onnx.py` | 各 EfficientNet-Lite 变体的 ONNX 导出脚本。 |
| `timm2onnx_local.py` | 转换流程中的本地 timm checkpoint 转 ONNX 辅助脚本。 |
| `get_calibration_data.py` | 转换流程中的校准图片预处理脚本。 |
| `x86_inference.py` | x86 参考推理脚本，用于 ONNX/HBIR/HBM 检查。 |

## 源模型

- 论文：`https://arxiv.org/abs/1905.11946`
- 源仓库：`https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet`
- EfficientNet-Lite 权重通过 timm 模型导出，例如 `timm/tf_efficientnet_lite0.in1k`。

重新生成 ONNX 时安装导出依赖：

```bash
pip install timm onnx
```

运行对应导出脚本，例如：

```bash
python3 get_efficientnet_lite0_onnx.py
```

## 校准数据

这些模型使用 100 张 ImageNet 验证集图片作为校准数据。使用以下命令生成 RGB 校准数据：

```bash
python3 get_calibration_data.py
```

YAML 中的校准数据路径为：

```text
./calibration_data_rgb
```

## 编译

使用对应 YAML 编译所需变体：

```bash
hb_compile --config efficientnet_lite0_config.yaml
```

运行示例默认下载已发布的 HBM 模型；只有修改源模型或转换配置时才需要重新编译。

## 模型配置

| 变体 | ONNX 名称 | HBM 输出前缀 | 输入 |
| --- | --- | --- | --- |
| Lite0 | `tf_efficientnet_lite0.onnx` | `efficientnet_lite0_224x224_nv12` | 224x224 |
| Lite1 | `tf_efficientnet_lite1.onnx` | `efficientnet_lite1_240x240_nv12` | 240x240 |
| Lite2 | `tf_efficientnet_lite2.onnx` | `efficientnet_lite2_260x260_nv12` | 260x260 |
| Lite3 | `tf_efficientnet_lite3.onnx` | `efficientnet_lite3_300x300_nv12` | 300x300 |
| Lite4 | `tf_efficientnet_lite4.onnx` | `efficientnet_lite4_380x380_nv12` | 380x380 |

所有变体均使用 NV12 运行时输入、RGB 训练输入、NCHW 训练布局、mean `127 127 127`、scale `0.007843 0.007843 0.007843`，march 为 `nash-e`。

## OE 工具链

模型转换请在 x86 Linux 主机的 RDK S100 OpenExplore 环境中完成，不建议在板端执行转换。

- OE 资源入口（docker+OE开发包）：<https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE 工具链在线手册：<https://toolchain.d-robotics.cc/>

请从 OE 资源入口获取适配 RDK S100/S100P 的 OpenExplore CPU Docker 镜像，并按实际文件名加载：

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
