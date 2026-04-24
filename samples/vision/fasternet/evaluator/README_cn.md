[English](./README.md) | 简体中文

# 模型评测

本目录记录 FasterNet sample 的评测说明和验证参考。

## 支持模型

| 模型 | 尺寸 | 类别数 |
| --- | --- | --- |
| FasterNet-S | 224x224 | 1000 |
| FasterNet-T2 | 224x224 | 1000 |
| FasterNet-T1 | 224x224 | 1000 |
| FasterNet-T0 | 224x224 | 1000 |

## 测试环境

- 平台：`RDK X5`
- 运行时后端：`hbm_runtime`
- 模型格式：`.bin`
- CPU：8xA55@1.8GHz，全核 Performance 调度
- BPU：1xBayes-e@1GHz，10TOPS 等效 INT8 算力

## 指标说明

- Float Top-1 为量化前浮点 ONNX 模型的 Top-1 精度。
- Quant Top-1 为量化部署模型的 Top-1 精度。
- Latency 为单帧、单线程、单 BPU 核心推理延迟。
- FPS 为多线程任务提交方式下的吞吐结果。

## Benchmark 结果

| 模型 | 尺寸 | 参数量 (M) | Float Top-1 | Quant Top-1 | 单线程延迟 (ms) | 多线程延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FasterNet-S | 224x224 | 31.1 | 77.04% | 76.15% | 6.73 | 24.34 | 162.83 |
| FasterNet-T2 | 224x224 | 15.0 | 76.50% | 76.05% | 3.39 | 11.56 | 342.48 |
| FasterNet-T1 | 224x224 | 7.6 | 74.29% | 71.25% | 1.96 | 5.58 | 708.40 |
| FasterNet-T0 | 224x224 | 3.9 | 71.75% | 68.50% | 1.41 | 3.48 | 1135.13 |

## 验证说明

本 sample 通过标准 Python 运行链路验证：

- `runtime/python/run.sh`
- `runtime/python/main.py`

示例会打印 Top-K 分类结果，并保存可视化图像。
