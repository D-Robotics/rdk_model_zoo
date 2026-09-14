[English](./README.md) | 简体中文

# 模型评测

本目录提供 VargConvNet sample 的验证说明。

## 支持模型

| 模型 | 输入尺寸 | 类别数 |
| --- | --- | --- |
| vargconvnet | 224x224 | 1000 |

## 测试环境

- 平台：`RDK X5`
- 运行后端：`hbm_runtime`
- 模型格式：`.bin`
- 输入格式：packed NV12

## Benchmark 结果

旧 VargConvNet demo 未提供公开 benchmark 表格。本 sample 保留 RDK X5 的运行验证流程和模型清单。

## 验证说明

本 sample 通过标准 Python 运行路径验证：

- `runtime/python/run.sh`
- `runtime/python/main.py`

程序会打印 Top-K 分类结果，并保存可视化图像。
