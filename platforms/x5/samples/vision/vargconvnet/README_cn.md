[English](./README.md) | 简体中文

# VargConvNet 模型说明

本目录给出 VargConvNet sample 在 Model Zoo 中的完整使用说明，包括算法概览、模型转换、运行时推理、模型文件管理和评测说明。

## 算法概述

VargConvNet 是面向边缘设备的轻量级卷积分类模型，用于 ImageNet-1k 图像分类。RDK X5 sample 提供 packed-NV12 `.bin` 模型和基于 `hbm_runtime` 的 Python 运行时。

### 算法功能

VargConvNet 支持以下任务：

- ImageNet 1000 类图像分类

### 算法特点

- 轻量级 CNN 分类模型。
- RDK X5 部署使用单输入 packed NV12。
- 输出为标准 ImageNet Top-K 分类结果。

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
|       |-- vargconvnet.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- box_turtle.JPEG
|   |-- ImageNet_1k.json
|   `-- inference.png
|-- README.md
`-- README_cn.md
```

## 快速体验

### Python

- Python 详细说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。
- 快速体验命令：

```bash
cd runtime/python
bash run.sh
```

## 模型转换

- 预编译 `.bin` 模型通过 [model](./model/README_cn.md) 目录提供。
- 转换说明请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 运行时推理

本 sample 当前维护的推理路径为 Python。

- Python 推理说明：[runtime/python/README_cn.md](./runtime/python/README_cn.md)

## 评测说明

评测说明和验证结果请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 性能数据

原始 VargConvNet demo 未提供公开 benchmark 表格。验证说明请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

![推理结果](./test_data/inference.png)

## License

遵循 Model Zoo 顶层 License。
