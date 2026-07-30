[English](./README.md) | 简体中文

# 模型转换

本目录提供 RepVGG sample 的转换侧参考资料。

## 概述

RepVGG 部署模型以 RDK X5 `.bin` 文件形式提供。本目录保留 OE 编译使用的 PTQ YAML 参考配置。

如果需要重新生成部署模型，请使用 OpenExplorer Docker 或对应 OE 包编译环境。

## 当前资产

本 sample 保留以下转换相关参考：

- 已发布部署模型：
  - `RepVGG_A0_224x224_nv12.bin`
  - `RepVGG_A1_224x224_nv12.bin`
  - `RepVGG_A2_224x224_nv12.bin`
  - `RepVGG_B0_224x224_nv12.bin`
  - `RepVGG_B1g2_224x224_nv12.bin`
  - `RepVGG_B1g4_224x224_nv12.bin`
- 运行时输入格式：packed NV12
- 运行时输出：ImageNet-1k 分类 logits
- PTQ 参考配置：
  - `RepVGG_A0_config.yaml`
  - `RepVGG_A1_config.yaml`
  - `RepVGG_A2_config.yaml`
  - `RepVGG_B0_config.yaml`
  - `RepVGG_B1g2_config.yaml`
  - `RepVGG_B1g4_config.yaml`

这些 YAML 文件是 OE/PTQ 编译参考配置，可在 OE 环境中配合 `hb_mapper checker` 和 `hb_mapper makertbin` 重新生成 RDK X5 部署模型。

## ONNX 导出参考

原始 RepVGG 流程从官方 RepVGG 实现导出 ONNX。导出参考如下：

1. 准备官方 RepVGG 源码和对应预训练权重。
2. 创建目标模型变体，例如 `create_RepVGG_B1g2(deploy=False)`。
3. 加载训练权重，并使用 `repvgg_model_convert()` 转换为部署结构。
4. 使用 `torch.onnx.export` 和 `1x3x224x224` dummy input 导出 ONNX。
5. 在 OE 环境中编译导出的 ONNX 模型。

## 转换说明

- 原始流程区分 `A0`、`A1`、`A2`、`B0`、`B1g2`、`B1g4` 多个变体。
- 本目录中的 YAML 文件与这些已发布变体对应。
- 重新生成目标 `.bin` 模型时，应选择匹配的 YAML 文件。

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
