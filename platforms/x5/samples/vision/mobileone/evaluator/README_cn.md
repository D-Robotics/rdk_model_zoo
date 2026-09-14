[English](./README.md) | 简体中文

# 模型评测

本目录提供 MobileOne sample 的 benchmark 说明和验证参考。

## 支持模型

| 模型 | 输入尺寸 | 类别数 |
| --- | --- | --- |
| MobileOne_S0 | 224x224 | 1000 |
| MobileOne_S1 | 224x224 | 1000 |
| MobileOne_S2 | 224x224 | 1000 |
| MobileOne_S3 | 224x224 | 1000 |
| MobileOne_S4 | 224x224 | 1000 |

## 测试环境

- 平台：`RDK X5`
- 运行时后端：`hbm_runtime`
- 模型格式：`.bin`
- CPU：8xA55@1.8GHz，全核 Performance 调度
- BPU：1xBayes-e@1GHz，等效 10TOPS INT8 算力

## 指标说明

- 浮点 Top-1 为量化前 ONNX 模型的分类精度。
- 量化 Top-1 为量化后部署模型的实际推理精度。
- 单线程时延为单帧、单线程、单 BPU 核的推理时延。
- 多线程时延为多线程任务提交场景下的测量结果。
- FPS 为 `RDK X5` 上的多线程吞吐测试结果。

## Benchmark 结果

| 模型 | 输入尺寸 | 参数量 (M) | 浮点 Top-1 | 量化 Top-1 | 单线程时延 (ms) | 多线程时延 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileOne_S4 | 224x224 | 14.8 | 78.75% | 76.50% | 4.58 | 15.44 | 256.52 |
| MobileOne_S3 | 224x224 | 10.1 | 77.27% | 75.75% | 2.93 | 9.04 | 437.85 |
| MobileOne_S2 | 224x224 | 7.8 | 74.75% | 71.25% | 2.11 | 6.04 | 653.68 |
| MobileOne_S1 | 224x224 | 4.8 | 72.31% | 70.45% | 1.31 | 3.69 | 1066.95 |
| MobileOne_S0 | 224x224 | 2.1 | 69.25% | 67.58% | 0.80 | 1.59 | 2453.17 |

## 验证说明

本 sample 通过标准 Python 运行链路进行验证：

- `runtime/python/run.sh`
- `runtime/python/main.py`

样例会输出 Top-K 分类结果，并保存可视化图像。
