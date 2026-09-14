[English](./README.md) | 简体中文

# 模型转换

本目录提供 EfficientViT sample 的转换侧参考资料。

## 概述

EfficientViT 的部署模型以 `RDK X5` `.bin` 文件形式提供。本目录保留用于 OE 编译的参考 PTQ YAML 配置。

如果需要重新生成部署模型，请使用 OpenExplorer Docker 或对应的 OE 包编译环境。

## 当前资产

本 sample 保留了以下转换相关内容：

- 已发布部署模型：
  - `EfficientViT_m5_224x224_nv12.bin`
- 运行时输入格式：packed NV12
- 运行时输出：ImageNet-1k 分类 logits
- 参考 PTQ 配置：
  - `EfficientViT_MSRA_m5_config.yaml`

本目录中的 YAML 文件是参考 OE/PTQ 编译配置，可在 OE 环境中配合 `hb_mapper checker` 和 `hb_mapper makertbin` 重新生成 `RDK X5` 部署模型。

## ONNX 导出参考

原始 EfficientViT_MSRA 流程基于 `timm` 的实现导出 ONNX，参考流程如下：

1. 安装 `timm`、`onnx`、`onnxsim` 等依赖。
2. 创建预训练的 `efficientvit_m5` 模型。
3. 使用 `1x3x224x224` 的 dummy input 通过 `torch.onnx.export` 导出模型。
4. 使用 `onnxsim` 对导出的 ONNX 进行简化。
5. 在 OE 环境中对简化后的 ONNX 进行编译。

## 转换说明

- EfficientViT_MSRA 内部包含 Softmax 节点。
- 在 OE/PTQ 流程中，这些 Softmax 节点需要通过 YAML 中的 `node_info` 配置固定在 BPU 上执行。
- 当前提供的 `EfficientViT_MSRA_m5_config.yaml` 已经包含所需的 Softmax 配置项。

## 转换参考

请结合 OE 包完成以下步骤：

- ONNX 准备
- PTQ 配置生成
- `hb_mapper checker`
- `hb_mapper makertbin`
- `hb_perf`
- `hrt_model_exec`

如果需要离线 Docker 镜像，也可以前往地瓜开发者社区获取：[https://forum.d-robotics.cc/t/topic/35229](https://forum.d-robotics.cc/t/topic/35229)。

## 输出协议

运行时 sample 默认使用以下协议：

- 输入张量：NV12 打包前为 `1x3x224x224`
- 输出张量：ImageNet-1k 分类 logits
