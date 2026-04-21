# 模型评估

本目录记录 MobileNetV3 sample 的评测说明与验证参考。

## 支持模型

| 模型 | 尺寸 | 类别数 |
| --- | --- | --- |
| MobileNetV3-Large | 224x224 | 1000 |

## 测试环境

- 平台：`RDK X5`
- 运行后端：`hbm_runtime`
- 模型格式：`.bin`

## Benchmark 结果

| 模型 | 浮点 Top-1 | 量化 Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- |
| MobileNetV3-Large | 74.8% | 64.8% | 2.02 | 714+ |

## 验证说明

本 sample 通过标准 Python 运行链路验证：

- `runtime/python/run.sh`
- `runtime/python/main.py`

示例会打印 Top-K 分类结果并保存可视化图像。
