English | [简体中文](./README_cn.md)

# FCOS 模型说明

本目录描述 FCOS 在本仓库中的标准化工作流，包括模型介绍、转换说明、运行推理和 benchmark 信息，面向 RDK X5 平台。

## 算法概述

FCOS 是经典的单阶段 anchor-free 检测算法，直接在特征图上预测类别分数、框回归和 center-ness，无需预先生成 anchor。

- **论文**: [Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf)
- **官方参考实现**: [tianzhi0549/FCOS](https://github.com/tianzhi0549/FCOS)

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

Model Zoo 已提供可直接使用的 `.bin` 模型。如需了解转换侧说明，请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 运行推理

当前示例提供 RDK X5 上的 Python runtime：

- 标准入口: [runtime/python/main.py](./runtime/python/main.py)
- 运行说明: [runtime/python/README_cn.md](./runtime/python/README_cn.md)

## 评测说明

Benchmark 和验证说明见 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 性能数据

| 模型 | 尺寸 | 类别数 | BPU 吞吐 | Python 后处理 |
| --- | --- | --- | --- | --- |
| `fcos_efficientnetb0` | 512x512 | 80 | `323.0 FPS` | `9 ms` |
| `fcos_efficientnetb2` | 768x768 | 80 | `70.9 FPS` | `16 ms` |
| `fcos_efficientnetb3` | 896x896 | 80 | `38.7 FPS` | `20 ms` |

![FCOS Demo](./test_data/demo_rdkx5_fcos_detect.jpg)

## License

遵循仓库顶层 License。
