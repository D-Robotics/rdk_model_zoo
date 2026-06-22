[English](./README.md) | 简体中文

# 模型转换

Model Zoo 已提供 MobileNetV1 的 S100 HBM 模型。只需要运行推理的用户可以
直接在 `../model/` 目录下载模型。

## 已发布产物

| 文件 | 输入 | 运行时 |
| --- | --- | --- |
| `mobilenetv1_224x224_nv12.hbm` | 224x224 NV12 (Y + UV) | `hbm_runtime` |

## 重新生成说明

MobileNetV1 使用 MobileNet-Caffe 模型作为源模型，并通过 RDK S100
OpenExplore 工具链完成转换。如需重新生成模型，请使用 S100 OE 包中的模型转换
环境，并保持当前运行时接口不变：两个 NV12 输入，分别为 Y plane 和 UV plane。

## 转换参考

- ONNX 导出
- PTQ 配置生成

## OE 资源入口

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
