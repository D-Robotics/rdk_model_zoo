[English](./README.md) | 简体中文

# 模型评测

本目录记录 RepViT sample 的评测说明和验证参考。

## 支持模型

| 模型 | 尺寸 | 类别数 |
| --- | --- | --- |
| RepViT-m0.9 | 224x224 | 1000 |
| RepViT-m1.0 | 224x224 | 1000 |
| RepViT-m1.1 | 224x224 | 1000 |

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
| RepViT-m1.1 | 224x224 | 8.2 | 77.73% | 77.50% | 2.32 | 6.69 | 590.42 |
| RepViT-m1.0 | 224x224 | 6.8 | 76.75% | 76.50% | 1.97 | 5.71 | 692.29 |
| RepViT-m0.9 | 224x224 | 5.1 | 76.32% | 75.75% | 1.65 | 4.37 | 902.69 |

## 验证说明

本 sample 通过标准 Python 运行链路验证：

- `runtime/python/run.sh`
- `runtime/python/main.py`

示例会打印 Top-K 分类结果，并保存可视化图像。
