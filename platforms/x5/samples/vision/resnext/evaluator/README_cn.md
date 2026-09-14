# 模型评测

本目录记录 ResNeXt sample 的评测说明和验证参考。

## 支持模型

| 模型 | 尺寸(像素) | 类别数 |
| --- | --- | --- |
| ResNeXt50_32x4d | 224x224 | 1000 |

## 测试环境

- 平台：`RDK X5`
- 运行时后端：`hbm_runtime`
- 模型格式：`.bin`
- CPU：8xA55@1.8GHz，全核心 Performance 调度
- BPU：1xBayes-e@1GHz，等效 10TOPS INT8 算力

## 指标说明

- 浮点 Top-1：模型量化前 ONNX 的评测精度。
- 量化 Top-1：量化部署模型的评测精度。
- 时延：单帧、单线程、单 BPU 核心下的理想前向时延。
- FPS：通过多线程持续提交任务以提升 BPU 吞吐得到的指标。

## Benchmark 结果

| 模型 | 尺寸(像素) | 参数量(M) | 浮点 Top-1 | 量化 Top-1 | 单线程时延(ms) | 多线程时延(ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNeXt50_32x4d | 224x224 | 24.99 | 76.25% | 76.00% | 5.89 | 20.90 | 189.61 |

## 验证总结

当前 sample 通过标准化 Python 运行路径进行验证：

- `runtime/python/run.sh`
- `runtime/python/main.py`

运行结果会打印 Top-K 分类结果，并保存可视化图像。
