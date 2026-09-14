[English](./README.md) | 简体中文

# 模型下载

本目录管理 Python 示例使用的 YOLOWorld 模型文件。

## 默认模型

| 模型 | 输入 | 格式 | 运行时 |
| --- | --- | --- | --- |
| `yolo_world.bin` | 图像 + 文本 embedding | `.bin` | `hbm_runtime` |

## 下载方式

```bash
bash download.sh
```

当本目录不存在模型文件时，脚本会从 RDK X5 Model Zoo 归档地址下载默认模型。
