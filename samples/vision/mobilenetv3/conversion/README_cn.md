# 模型转换

本目录记录 MobileNetV3 sample 的转换侧说明。

## 说明

当前仓库不维护完整的 MobileNetV3 转换流程。  
如果需要重新生成部署模型，请直接参考 OE 包中的完整转换流程。

## 当前保留内容

本 sample 当前保留的转换相关信息包括：

- 已发布部署模型名称：`MobileNetV3_224x224_nv12.bin`
- 运行时输入格式：packed NV12
- 运行时输出：ImageNet-1k 分类 logits
- 参考 PTQ 配置：`MobileNetV3_config.yaml`

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
