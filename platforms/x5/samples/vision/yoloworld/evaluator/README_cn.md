[English](./README.md) | 简体中文

# 模型评测

本目录提供 YOLOWorld sample 的验证说明。

## 支持模型

| 模型 | 任务 | 输入尺寸 | 词表槽位 |
| --- | --- | --- | --- |
| `yolo_world.bin` | 开放词表目标检测 | 640x640 | 32 |

## 测试环境

- 平台：`RDK X5`
- 运行时后端：`hbm_runtime`
- 模型格式：`.bin`
- 默认图片：`test_data/dog.jpeg`
- 默认 prompt：`dog`

## Benchmark 结果

原始 YOLOWorld demo 未提供公开 benchmark 表。当前 sample 保留 RDK X5 的运行验证流程和模型协议说明。

## 验证说明

本 sample 通过标准 Python 运行链路验证：

- `runtime/python/run.sh`
- `runtime/python/main.py`

默认行为预期为检测测试图片中的狗，并保存可视化结果图。
