[English](./README.md) | 简体中文

# 模型评估

本目录记录 GoogLeNet sample 的评测说明与验证参考。

## 支持模型

| 模型 | 尺寸 | 类别数 |
| --- | --- | --- |
| GoogLeNet | 224x224 | 1000 |

## 测试环境

- 平台：`RDK X5`
- 运行后端：`hbm_runtime`
- 模型格式：`.bin`
- CPU：8xA55@1.8GHz，全核心 Performance 调度
- BPU：1xBayes-e@1GHz，共 10TOPS 等效 INT8 算力

## 指标说明

- 浮点 Top-1 使用模型量化前 ONNX 的推理精度。
- 量化 Top-1 使用量化后部署模型的实际推理精度。
- 延迟为单帧、单线程、单 BPU 核心的推理延迟。
- FPS 为多线程向 BPU 提交任务时的吞吐数据，用于反映模型吞吐能力。

## Benchmark 结果

| 模型 | 尺寸 | 参数量 (M) | 浮点 Top-1 | 量化 Top-1 | 单线程延迟 (ms) | 多线程延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GoogLeNet | 224x224 | 6.81 | 68.72% | 67.71% | 2.19 | 6.30 | 626.27 |

## 验证说明

本 sample 通过标准 Python 运行链路验证：

- `runtime/python/run.sh`
- `runtime/python/main.py`

示例会打印 Top-K 分类结果并保存可视化图像。
