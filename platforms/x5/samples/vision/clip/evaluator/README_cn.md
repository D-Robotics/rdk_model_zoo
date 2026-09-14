[English](./README.md) | 简体中文

# 模型评测

本目录提供 CLIP 图文匹配 sample 的验证说明。

## 支持模型

| 模型 | 作用 | 运行时 |
| --- | --- | --- |
| `img_encoder.bin` | 图像 encoder | `hbm_runtime` |
| `text_encoder.onnx` | 文本 encoder | `onnxruntime` |

## 测试环境

- 平台：`RDK X5`
- 图像 encoder 格式：`.bin`
- 文本 encoder 格式：`.onnx`
- Python 依赖：`onnxruntime`、`ftfy`、`regex`
- 默认图片：`test_data/dog.jpg`
- 默认文本：`a diagram`、`a dog`

## Benchmark 结果

原始 CLIP demo 未提供公开 benchmark 表。当前 sample 保留 RDK X5 的运行验证流程和模型协议说明。

## 验证说明

本 sample 通过标准 Python 运行链路验证：

- `runtime/python/run.sh`
- `runtime/python/main.py`

默认行为预期为：狗图片对 `a dog` 的相似度高于 `a diagram`。
