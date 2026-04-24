# 模型转换

本目录整理 ResNeXt sample 的转换侧说明。

## 概述

ResNeXt 的部署模型以 RDK X5 `.bin` 文件形式提供。本目录保留了 OE 编译所需的参考 PTQ YAML 配置。

如需重新生成部署模型，请使用 OpenExplorer Docker 或对应的 OE 包编译环境。

## 当前资产

本 sample 当前保留以下转换参考资产：

- 已发布部署模型：
  - `ResNeXt50_32x4d_224x224_nv12.bin`
- 运行时输入格式：packed NV12
- 运行时输出：ImageNet-1k 分类 logits
- 参考 PTQ 配置：
  - `ResNeXt50_32x4d_config.yaml`

本目录中的 YAML 文件是参考 OE/PTQ 编译配置，可在 OE 环境中结合 `hb_mapper checker` 与 `hb_mapper makertbin` 重新生成 RDK X5 部署模型。

## ONNX 导出参考

原始 ResNeXt 流程使用 `timm` 中的模型实现导出 ONNX，导出流程为：

1. 安装 `timm`、`onnx`、`onnxsim` 等 Python 包。
2. 创建带有预训练权重的 `resnext50_32x4d` 模型。
3. 使用 `1x3x224x224` dummy input 通过 `torch.onnx.export` 导出 ONNX。
4. 使用 `onnxsim` 对 ONNX 模型进行简化。
5. 在 OE 环境中编译简化后的 ONNX 模型。

## 转换参考

如需重新转换，请参考 OE 包完成：

- ONNX 准备
- PTQ 配置生成
- `hb_mapper checker`
- `hb_mapper makertbin`
- `hb_perf`
- `hrt_model_exec`

也可以前往地瓜开发者社区获取离线版本的 Docker 镜像: [https://forum.d-robotics.cc/t/topic/28035](https://forum.d-robotics.cc/t/topic/28035)

## 输出协议

当前 runtime sample 默认：

- 输入张量形状：打包成 NV12 前为 `1x3x224x224`
- 输出张量：ImageNet-1k 分类 logits
