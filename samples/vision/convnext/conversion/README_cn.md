# 模型转换

本目录介绍如何将 ConvNeXt 模型转换为可在 BPU 上运行的 BIN/HBM 格式。

## 目录结构

```text
.
├── ConvNeXt_atto.yaml              # Atto 模型的 PTQ 配置文件
├── ConvNeXt_femto.yaml             # Femto 模型的 PTQ 配置文件
├── ConvNeXt_nano.yaml              # Nano 模型的 PTQ 配置文件
├── ConvNeXt_pico.yaml              # Pico 模型的 PTQ 配置文件
├── README.md                        # 使用说明 (英文)
└── README_cn.md                     # 使用说明 (中文)
```

## 转换流程

1. **准备环境**: 安装 RDK X5 OpenExplore 工具链。
2. **导出 ONNX**: 将预训练模型导出为 ONNX 格式。
3. **PTQ 量化**: 使用 `hb_mapper` 配合提供的 YAML 配置文件进行转换。

有关通用转换流程的更多详细信息，请参考 [Model Zoo 转换指南](../../../../../docs/README.md)。
