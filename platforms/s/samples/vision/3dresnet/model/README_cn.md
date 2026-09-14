[English](./README.md) | 简体中文

# 模型文件

本目录提供 3D ResNet-18 视频动作分类 sample 的模型下载脚本。

## 下载

```bash
bash download_model.sh s100
```

模型会下载到：

```text
model/s100/r3d_18.hbm
```

## 模型说明

| 文件 | 说明 |
| ---- | ---- |
| `s100/r3d_18.hbm` | R3D-18 HBM 模型，输入为 `(1, 3, 16, 112, 112)` float32 视频片段，输出为 Kinetics-400 logits。 |
