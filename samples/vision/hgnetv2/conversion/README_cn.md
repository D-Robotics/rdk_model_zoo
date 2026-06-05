# HGNetv2 模型转换与编译指南

[English](./README.md) | 简体中文

本目录提供了将 HGNetv2 模型转换为适配地瓜机器人（D-Robotics）RDK 硬件的 BPU 量化模型（`.bin`）的工具与说明。

## 模型编译环境

为了转换模型，您需要安装 **RDK X5 OpenExplore 工具链**。

### Docker 安装

**RDK X5 OpenExplore 1.2.8**
```bash
docker pull openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
```
或者前往地瓜开发者社区获取离线版本的 Docker 镜像: [https://forum.d-robotics.cc/t/topic/28035](https://forum.d-robotics.cc/t/topic/28035)

**启动容器**:
```bash
# 挂载您的 model zoo 目录到容器中
docker run -it --rm -v /path/to/rdk_model_zoo:/data openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```
---

## 转换流程

### 1. pth 转 onnx 模型

我们提供了 `onnx_export/export_hgnetv2_b0_bpu.py` 脚本，可以将 pth 文件转为 onnx 文件。

### 2. onnx 转 bin 模型

**准备工作**:
- 已经导出为 BPU 适配的 ONNX 模型（参考 `onnx_export/export_hgnetv2_b0_bpu.py`）。
- 准备一个文件夹，包含 20~50 张用于量化校准的图片（`.jpg` 或 `.png`）。

**运行转换**:
```bash
hb_mapper makertbin --model-type onnx --config hgnetv2_b0.yaml
```
转换成功后，生成的 `.bin` 模型文件将位于 ONNX 模型的同级目录下。

---

## License
本目录下的工具遵循 [Apache 2.0 License](../../../../LICENSE)。