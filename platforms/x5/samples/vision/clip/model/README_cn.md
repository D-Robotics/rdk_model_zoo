[English](./README.md) | 简体中文

# 模型下载

本目录管理 Python 示例使用的 CLIP 模型文件。

## 默认模型

| 模型 | 作用 | 格式 | 运行时 |
| --- | --- | --- | --- |
| `img_encoder.bin` | 图像 encoder | `.bin` | `hbm_runtime` |
| `text_encoder.onnx` | 文本 encoder | `.onnx` | `onnxruntime` |

## 下载方式

```bash
bash download.sh
```

当本目录不存在模型文件时，脚本会从 RDK X5 Model Zoo 归档地址下载两个默认模型。
