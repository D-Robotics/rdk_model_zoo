# 模型评测

本目录记录 MobileNetV2 sample 的评测说明和验证参考。

## 支持模型

| 模型 | 尺寸 | 类别数 |
| --- | --- | --- |
| MobileNetV2 | 224x224 | 1000 |

## 测试环境

- 平台：`RDK X5`
- 运行后端：`hbm_runtime`
- 模型格式：`.bin`

## Benchmark 结果

| 模型 | 浮点 Top-1 | 量化 Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- |
| MobileNetV2 | 72.0% | 68.17% | 1.42 | 1152.07 |

## 验证说明

当前 sample 通过标准 Python 运行路径验证：

- `runtime/python/run.sh`
- `runtime/python/main.py`

sample 会输出 Top-K 分类结果，并保存可视化图片。
