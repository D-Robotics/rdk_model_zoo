[English](./README.md) | 简体中文

# 模型转换

本目录记录 RepViT sample 的转换侧说明。

## 概述

RepViT 部署模型以 RDK X5 `.bin` 文件形式提供。本目录保留 OE 编译使用的 PTQ YAML 参考配置。

如需重新生成部署模型，请使用 OpenExplorer Docker 或对应 OE 包编译环境。

## 当前资产

本 sample 保留以下转换相关参考：

- 已发布部署模型：
  - `RepViT_m0_9_224x224_nv12.bin`
  - `RepViT_m1_0_224x224_nv12.bin`
  - `RepViT_m1_1_224x224_nv12.bin`
- 运行时输入格式：packed NV12
- 运行时输出：ImageNet-1k 分类 logits
- PTQ 参考配置：
  - `RepViT_m0_9_config.yaml`
  - `RepViT_m1_0_config.yaml`
  - `RepViT_m1_1_config.yaml`

本目录下的 YAML 文件是 OE/PTQ 编译参考配置，可在 OE 环境中配合 `hb_mapper checker` 和 `hb_mapper makertbin` 重新生成 RDK X5 部署模型。

## ONNX 导出参考

原 RepViT 流程使用 `timm` 导出 ONNX 模型，流程如下：

1. 安装 `timm`、`onnx`、`onnxsim` 等 Python 依赖。
2. 通过 `timm.models.create_model` 创建目标 RepViT 模型，例如 `repvit_m0_9`、`repvit_m1_0` 或 `repvit_m1_1`。
3. 使用 `torch.onnx.export` 导出模型，dummy input 为 `1x3x224x224`。
4. 使用 `onnxsim.simplify` 简化 ONNX 模型。
5. 在 OE 环境中编译简化后的 ONNX 模型。

## 转换说明

请参考 OE 包完成：

- ONNX 准备
- PTQ 配置生成
- `hb_mapper checker`
- `hb_mapper makertbin`
- `hb_perf`
- `hrt_model_exec`

也可以前往地瓜开发者社区获取离线版本的 Docker 镜像：[https://forum.d-robotics.cc/t/topic/28035](https://forum.d-robotics.cc/t/topic/28035)。

## 输出协议

运行时 sample 默认使用：

- NV12 打包前输入张量形状：`1x3x224x224`
- 输出张量：ImageNet-1k 分类 logits
