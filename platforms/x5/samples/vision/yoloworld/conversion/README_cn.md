[English](./README.md) | 简体中文

# 模型转换

本目录提供 YOLOWorld sample 的转换侧说明。

## 概述

部署模型为 `yolo_world.bin`，面向 RDK X5 编译。原始 demo 提供已编译模型和离线词表 embedding JSON，但仓库中未提供转换 YAML 或导出脚本。

## 模型协议

### 输入

| 输入 | 数据类型 | 形状 | 布局 |
| --- | --- | --- | --- |
| image | FP32 | `1 x 3 x 640 x 640` | NCHW |
| text | FP32 | `1 x 32 x 512 x 1` | NCHW-like embedding tensor |

### 输出

| 输出 | 数据类型 | 形状 |
| --- | --- | --- |
| classes_score | FP32 | `1 x 8400 x 32` |
| bboxes | FP32 | `1 x 8400 x 4` |

## 转换参考

如需重新生成 `.bin` 模型，请使用 OpenExplorer Docker 或对应 OE 包编译环境。

Docker 离线镜像也可以前往地瓜开发者社区获取：[https://forum.d-robotics.cc/t/topic/35229](https://forum.d-robotics.cc/t/topic/35229)。
