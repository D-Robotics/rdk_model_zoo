[English](./README.md) | 简体中文

# 模型评测

本目录记录 FastViT sample 的评测说明和验证参考。

## 支持模型

| 模型 | 尺寸 | 类别数 |
| --- | --- | --- |
| FastViT-SA12 | 224x224 | 1000 |
| FastViT-S12 | 224x224 | 1000 |
| FastViT-T12 | 224x224 | 1000 |
| FastViT-T8 | 224x224 | 1000 |

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
| FastViT-SA12 | 224x224 | 10.9 | 78.25% | 74.50% | 11.56 | 42.45 | 93.44 |
| FastViT-S12 | 224x224 | 8.8 | 76.50% | 72.00% | 5.86 | 20.45 | 193.87 |
| FastViT-T12 | 224x224 | 6.8 | 74.75% | 70.43% | 4.97 | 16.87 | 234.78 |
| FastViT-T8 | 224x224 | 3.6 | 73.50% | 68.50% | 2.09 | 5.93 | 667.21 |

## 验证说明

本 sample 通过标准 Python 运行链路验证：

- `runtime/python/run.sh`
- `runtime/python/main.py`

示例会打印 Top-K 分类结果，并保存可视化图像。
