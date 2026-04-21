[English](./README.md) | 简体中文

# 模型转换

本目录记录 GoogLeNet sample 的转换侧说明。

## 说明

GoogLeNet 已发布的部署模型以 RDK X5 `.bin` 文件提供。本目录不维护源 ONNX 模型和独立转换脚本。

如需重新生成部署模型，请使用 OpenExplorer Docker 或对应 OE 包编译环境。

## 当前保留内容

本 sample 当前保留的转换相关信息包括：

- 已发布部署模型：
  - `googlenet_224x224_nv12.bin`
- 运行时输入格式：packed NV12
- 运行时输出：ImageNet-1k 分类 logits

## 转换参考

以下内容请参考 OE 包：

- ONNX 准备
- PTQ 配置生成
- `hb_mapper checker`
- `hb_mapper makertbin`
- `hb_perf`
- `hrt_model_exec`

Docker 镜像也可以通过地瓜开发者社区离线获取：[https://forum.d-robotics.cc/t/topic/28035](https://forum.d-robotics.cc/t/topic/28035)。

## 输出协议

当前 runtime sample 默认协议如下：

- 输入张量形状：`1x3x224x224`，随后打包为 NV12
- 输出张量：ImageNet-1k 分类 logits
