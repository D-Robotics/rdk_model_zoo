[English](./README.md) | 简体中文

# CLIP 模型说明

本目录给出 CLIP image-text matching sample 在 Model Zoo 中的完整使用说明，包括算法概览、模型转换说明、运行时推理、模型文件管理和评测说明。

## 算法概览

CLIP 将图像和文本映射到共享特征空间，并通过余弦相似度完成匹配。本 sample 保留原始 demo 的资产边界：图像 encoder 为 RDK X5 `.bin` 模型，文本 encoder 为 ONNX 模型。

### 算法功能

本 sample 支持以下任务：

- 图文相似度匹配

### 算法特点

- 图像 encoder 通过 `hbm_runtime` 在 BPU 上执行。
- 文本 encoder 通过 `onnxruntime` 执行。
- 使用 NumPy 预处理和 tokenizer，去除 notebook 运行依赖。

## 目录结构

```text
.
|-- conversion
|   |-- README.md
|   `-- README_cn.md
|-- evaluator
|   |-- README.md
|   `-- README_cn.md
|-- model
|   |-- download.sh
|   |-- README.md
|   `-- README_cn.md
|-- runtime
|   `-- python
|       |-- bpe_simple_vocab_16e6.txt.gz
|       |-- clip_retrieval.py
|       |-- main.py
|       |-- README.md
|       |-- README_cn.md
|       |-- run.sh
|       `-- simple_tokenizer.py
|-- test_data
|   |-- dog.jpg
|   `-- inference.png
|-- README.md
`-- README_cn.md
```

## 快速开始

### Python

- 详细 Python 使用说明见 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。
- 快速运行：

```bash
cd runtime/python
bash run.sh
```

## 模型转换

- 模型文件由 [model](./model/README_cn.md) 目录提供。
- 转换侧说明见 [conversion/README_cn.md](./conversion/README_cn.md)。

## 运行时推理

本 sample 维护的推理入口为 Python。

- Python 运行说明：[runtime/python/README_cn.md](./runtime/python/README_cn.md)

## 评测说明

评测说明和验证记录见 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 性能数据

原始 CLIP demo 未提供公开 benchmark 表。验证说明见 [evaluator/README_cn.md](./evaluator/README_cn.md)。

![推理结果](./test_data/inference.png)

## 许可证

遵循 Model Zoo 顶层 License。
