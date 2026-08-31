[English](./README.md) | 简体中文

# DINOv2 模型下载指南

## 下载命令

模型 march 由板端 SoC 自动检测，在目标板上执行：

```bash
cd samples/vision/dinov2/model
bash download_model.sh
```

在任意机器上下载指定 march：

```bash
bash download_model.sh nash-e   # RDK S100
bash download_model.sh nash-m   # RDK S100P
bash download_model.sh nash-p   # RDK S600
```

## 模型列表

| 模型名 | 支持的 BPU |
|---|---|
| `dinov2_vits14_224_int16_nashe.hbm` | Nash-E |
| `dinov2_vits14_224_int16_nashm.hbm` | Nash-M |
| `dinov2_vits14_224_int16_nashp.hbm` | Nash-P |

## 许可

本模型由 Meta AI 发布的 Apache-2.0 许可
[DINOv2](https://github.com/facebookresearch/dinov2) 权重量化而来。见
[../../../../LICENSE](../../../../LICENSE)。
