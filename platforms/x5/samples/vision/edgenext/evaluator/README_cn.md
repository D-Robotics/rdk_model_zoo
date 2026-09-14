[English](./README.md) | 简体中文

# 模型评测

本目录记录 EdgeNeXt sample 的评测说明和验证参考。

## 支持模型

| 模型 | 尺寸 | 类别数 |
| --- | --- | --- |
| EdgeNeXt-base | 224x224 | 1000 |
| EdgeNeXt-small | 224x224 | 1000 |
| EdgeNeXt-x-small | 224x224 | 1000 |
| EdgeNeXt-xx-small | 224x224 | 1000 |

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
| EdgeNeXt-base | 224x224 | 18.51 | 78.21% | 74.52% | 8.80 | 32.31 | 113.35 |
| EdgeNeXt-small | 224x224 | 5.59 | 76.50% | 71.75% | 4.41 | 14.93 | 226.15 |
| EdgeNeXt-x-small | 224x224 | 2.34 | 71.75% | 66.25% | 2.88 | 9.63 | 345.73 |
| EdgeNeXt-xx-small | 224x224 | 1.33 | 69.50% | 64.25% | 2.47 | 7.24 | 403.49 |

## 验证说明

本 sample 通过标准 Python 运行链路验证：

- `runtime/python/run.sh`
- `runtime/python/main.py`

示例会打印 Top-K 分类结果，并保存可视化图像。
