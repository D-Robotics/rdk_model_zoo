[English](./README.md) | 简体中文

# 模型转换

本目录提供 MobileOne sample 的转换侧参考资料。

## 概述

MobileOne 的部署模型以 `RDK X5` `.bin` 文件形式提供。本目录保留用于 OE 编译的参考 PTQ YAML 配置。

如果需要重新生成部署模型，请使用 OpenExplorer Docker 或对应的 OE 包编译环境。

## 当前资产

本 sample 保留了以下转换相关内容：

- 已发布部署模型：
  - `MobileOne_S0_224x224_nv12.bin`
  - `MobileOne_S1_224x224_nv12.bin`
  - `MobileOne_S2_224x224_nv12.bin`
  - `MobileOne_S3_224x224_nv12.bin`
  - `MobileOne_S4_224x224_nv12.bin`
- 运行时输入格式：packed NV12
- 运行时输出：ImageNet-1k 分类 logits
- 参考 PTQ 配置：
  - `MobileOne_S0_config.yaml`
  - `MobileOne_S1_config.yaml`
  - `MobileOne_S2_config.yaml`
  - `MobileOne_S3_config.yaml`
  - `MobileOne_S4_config.yaml`

这些 YAML 文件是参考 OE/PTQ 编译配置，可在 OE 环境中配合 `hb_mapper checker` 和 `hb_mapper makertbin` 重新生成 `RDK X5` 部署模型。

## ONNX 导出参考

原始 MobileOne 流程基于官方仓库导出 ONNX，参考流程如下：

1. 安装 `torch`、`onnx`、`onnxsim` 等依赖。
2. 克隆官方 `apple/ml-mobileone` 仓库。
3. 加载目标变体的 unfused 权重，例如 `mobileone_s0_unfused.pth.tar`。
4. 创建对应的 MobileOne 模型，并执行 `reparameterize_model(model)`。
5. 使用 `1x3x224x224` 的 dummy input 通过 `torch.onnx.export` 导出模型。
6. 使用 `onnxsim` 对导出的 ONNX 进行简化。
7. 在 OE 环境中对简化后的 ONNX 进行编译。

## 转换说明

- 原始流程区分 `S0` 到 `S4` 多个变体。
- 本目录中的 YAML 文件与这些已发布变体一一对应。
- 重新生成 `.bin` 模型时，请选择与目标模型一致的 YAML 文件。

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
