# 模型转换

本目录记录 MobileNetV4 sample 的转换侧说明。

## 说明

当前仓库不维护完整的 MobileNetV4 转换流程。  
如果需要重新生成部署模型，请直接参考 OE 包中的完整转换流程。

## 当前保留内容

本 sample 当前保留的转换相关信息包括：

- 已发布部署模型：
  - `MobileNetV4_conv_small_224x224_nv12.bin`
  - `MobileNetV4_conv_medium_224x224_nv12.bin`
- 运行时输入格式：packed NV12
- 运行时输出：ImageNet-1k 分类 logits
- 参考 PTQ 配置：
  - `MobileNetV4_small.yaml`
  - `MobileNetV4_medium.yaml`

本目录中的 YAML 文件为 OE/PTQ 编译配置参考，可在 OE 环境中配合
`hb_mapper checker` 和 `hb_mapper makertbin` 重新生成 RDK X5 部署模型。

## 转换参考

以下内容请直接参考 OE 包：

- ONNX 导出
- PTQ 配置生成
- `hb_mapper checker`
- `hb_mapper makertbin`
- `hb_perf`
- `hrt_model_exec`

## 输出协议

当前 runtime sample 默认协议如下：

- 输入张量形状：`1x3x224x224`，随后打包为 NV12
- 输出张量：ImageNet-1k 分类 logits
