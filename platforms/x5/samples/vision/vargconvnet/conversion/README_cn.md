[English](./README.md) | 简体中文

# 模型转换

本目录提供 VargConvNet sample 的转换侧说明。

## 概述

VargConvNet 部署模型以 RDK X5 `.bin` 文件形式提供。旧 demo 未包含 PTQ YAML 文件或独立 ONNX 导出脚本，因此本目录记录部署协议和 OE 参考流程。

如果需要重新生成部署模型，请使用 OpenExplorer Docker 或对应 OE 包编译环境。

## 当前资产

- 已发布部署模型：
  - `vargconvnet_224x224_nv12.bin`
- 运行时输入格式：packed NV12
- 运行时输出：ImageNet-1k 分类 logits

## 转换参考

请参考 OE 包完成以下流程：

- ONNX 准备
- PTQ 配置生成
- `hb_mapper checker`
- `hb_mapper makertbin`
- `hb_perf`
- `hrt_model_exec`

也可以前往地瓜开发者社区获取离线版本的 Docker 镜像：[https://forum.d-robotics.cc/t/topic/35229](https://forum.d-robotics.cc/t/topic/35229)。

## 输出协议

运行时 sample 假设：

- 输入 tensor 在 NV12 packing 前为 `1x3x224x224`
- 输出 tensor 为 ImageNet-1k 分类 logits
