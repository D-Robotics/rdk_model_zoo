[English](./README.md) | 简体中文

# 模型转换

本目录记录 EfficientNet sample 的转换侧说明。

## 说明

EfficientNet 部署模型以 RDK X5 `.bin` 文件提供。本目录保留用于 OE 编译的参考 PTQ YAML 文件。

如需重新生成部署模型，请使用 OpenExplorer Docker 或对应 OE 包编译环境。

## 当前保留内容

本 sample 当前保留的转换相关信息包括：

- 已发布部署模型：
  - `EfficientNet_B2_224x224_nv12.bin`
  - `EfficientNet_B3_224x224_nv12.bin`
  - `EfficientNet_B4_224x224_nv12.bin`
- 运行时输入格式：packed NV12
- 运行时输出：ImageNet-1k 分类 logits
- 参考 PTQ 配置：
  - `EfficientNet_B2_config.yaml`
  - `EfficientNet_B3_config.yaml`
  - `EfficientNet_B4_config.yaml`

本目录中的 YAML 文件为 OE/PTQ 编译配置参考，可在 OE 环境中配合 `hb_mapper checker` 和 `hb_mapper makertbin` 重新生成 RDK X5 部署模型。

## ONNX 导出参考

原始 demo 使用 `timm` 导出 EfficientNet ONNX 模型。导出流程为：

1. 使用 `timm.models.create_model` 创建目标 EfficientNet 模型。
2. 使用 `torch.onnx.export` 导出 ONNX。
3. 使用 `onnxsim.simplify` 简化 ONNX。
4. 在 OE 环境中编译简化后的 ONNX 模型。

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
