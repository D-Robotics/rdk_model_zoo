[English](./README.md) | 简体中文

# LPRNet 模型说明

本目录提供面向 RDK X5 的标准化 LPRNet 示例，包括模型介绍、运行推理与 benchmark 说明。

## 算法概述

LPRNet 是一种轻量级端到端车牌识别网络，输入车牌裁剪图后可直接输出字符序列，无需显式字符分割。

- **论文**: [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447)

## 目录结构

```text
.
├── conversion
├── evaluator
├── model
├── runtime
├── test_data
├── README.md
└── README_cn.md
```

## 快速开始

```bash
cd runtime/python
bash run.sh
```

运行细节请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

## 模型转换

本示例已提供可直接运行的 `.bin` 模型。转换侧说明请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 运行推理

当前示例提供 RDK X5 上的 Python runtime。

- 标准入口: [runtime/python/main.py](./runtime/python/main.py)
- 运行说明: [runtime/python/README_cn.md](./runtime/python/README_cn.md)

## 评测说明

Benchmark 与验证说明见 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 性能数据

| 模型 | 输入尺寸 | 输入类型 | 板端性能 |
| --- | --- | --- | --- |
| `lpr.bin` | `1x3x24x94` | `float32` 二进制张量 | `266 FPS / 3.75 ms` |

参考车牌图片：

![LPRNet Example](./test_data/example.jpg)

## License

遵循仓库顶层 License。
