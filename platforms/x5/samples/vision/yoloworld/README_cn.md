[English](./README.md) | 简体中文

# YOLOWorld 模型说明

本目录给出 YOLOWorld open-vocabulary detection sample 在 Model Zoo 中的完整使用说明，包括算法概览、模型转换说明、运行时推理、模型文件管理和评测说明。

## 算法概览

YOLOWorld 在 YOLO 目标检测基础上引入开放词表 prompt 能力。本 sample 使用离线词表 embedding 和 RDK X5 `.bin` 模型，在运行时无需文本 encoder 即可检测用户指定类别。

### 算法功能

本 sample 支持以下任务：

- 基于离线文本 embedding 的开放词表目标检测

### 算法特点

- 双输入模型：图像 tensor 和 32 槽位文本 embedding tensor。
- 离线词表 embedding 以 JSON 文件保存。
- Python wrapper 标准返回 boxes、scores 和 class IDs。

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
|       |-- main.py
|       |-- yoloworld_det.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- dog.jpeg
|   |-- inference.png
|   `-- offline_vocabulary_embeddings.json
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

- 预编译 `.bin` 模型由 [model](./model/README_cn.md) 目录提供。
- 转换侧说明见 [conversion/README_cn.md](./conversion/README_cn.md)。

## 运行时推理

本 sample 维护的推理入口为 Python。

- Python 运行说明：[runtime/python/README_cn.md](./runtime/python/README_cn.md)

## 评测说明

评测说明和验证记录见 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 性能数据

原始 YOLOWorld demo 未提供公开 benchmark 表。验证说明见 [evaluator/README_cn.md](./evaluator/README_cn.md)。

![推理结果](./test_data/inference.png)

## 许可证

遵循 Model Zoo 顶层 License。
