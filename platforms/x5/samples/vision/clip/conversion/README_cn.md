[English](./README.md) | 简体中文

# 模型转换

本目录提供 CLIP sample 的转换侧说明。

## 概述

CLIP sample 使用两个部署资产：

- `img_encoder.bin`：RDK X5 BPU 图像 encoder。
- `text_encoder.onnx`：通过 `onnxruntime` 在 CPU 上执行的 ONNX 文本 encoder。

原始 demo 未在仓库中提供转换 YAML 或导出脚本。因此当前维护的 sample 记录模型协议，并使用 Model Zoo 归档中发布的模型文件。

## 模型协议

### 图像 Encoder

| 输入 | 数据类型 | 形状 | 布局 |
| --- | --- | --- | --- |
| image | FP32 | `1 x 3 x 224 x 224` | NCHW |

| 输出 | 数据类型 | 形状 |
| --- | --- | --- |
| image_feature | FP32 | `1 x 512` |

### 文本 Encoder

| 输入 | 数据类型 | 形状 |
| --- | --- | --- |
| texts | INT32 | `num_text x 77` |

| 输出 | 数据类型 | 形状 |
| --- | --- | --- |
| text_features | FP32 | `num_text x 512` |

## 转换参考

如需重新生成图像 encoder `.bin`，请使用 OpenExplorer Docker 或对应 OE 包编译环境。文本 encoder 在本 sample 中仍作为 ONNX runtime 资产保留。

Docker 离线镜像也可以前往地瓜开发者社区获取：[https://forum.d-robotics.cc/t/topic/35229](https://forum.d-robotics.cc/t/topic/35229)。
