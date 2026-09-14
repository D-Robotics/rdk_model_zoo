[English](./README.md) | 简体中文

# DINOv2 模型说明

DINOv2 是 Meta AI 发表的自监督视觉 transformer 骨干网络（Oquab 等，
2023）。本样例将 ViT-S/14 变体以 int16 量化部署到 RDK S100/S100P/S600，
作为通用图像 embedding 模型，是 RDK 模型仓中首个自监督基础模型样例。

## 算法概述

DINOv2 是一个标准 ViT：patch-14 卷积 stem、12 个 pre-LN transformer
block（融合 qkv 注意力 + GELU MLP + LayerScale）、最终 LayerNorm。部署图
将 SDPA 注意力改写为显式 MatMul + Softmax，并将插值后的位置编码烘焙为
常量，使整网编译为纯 BPU 算子、零 CPU 回退。

## 算法能力

- 全局图像 embedding：`cls_feat`，形状 `(1, 384)`。
- 稠密 patch 级特征：`patch_feat`，形状 `(1, 256, 384)`。
- 下游用途：图像检索/相似度、linear-probe 分类，以及作为稠密任务的
  冻结骨干（同一编码器家族驱动 Depth Anything V2）。

## 算法特性

- int16 PTQ + 默认（KL）校准：板端执行的 cosine 范围为 `cls_feat`
  0.9987-0.9989、`patch_feat` 0.9975-0.9986（见
  [evaluator](./evaluator/README_cn.md)）。
- 100% BPU 执行，无 CPU 算子回退（nash-e 上 800/800 算子）。
- 每个 march 一份 `.hbm`（Nash-E / Nash-M / Nash-P），板端自动选择。

## 目录结构

```text
dinov2/
├── conversion                     # ONNX 导出 + PTQ 转换流程
│   ├── onnx_export/               # PyTorch -> ONNX 导出脚本
│   └── mapper.py                  # 一键转换入口
├── evaluator                      # 实测性能 / 精度记录
├── model                          # download_model.sh + 模型列表
├── runtime                        # 板端推理演示
│   └── python                     # 基于 hbm_runtime 的 python runtime
└── test_data                      # 演示图片
```

## 快速开始

```bash
# 在板端：
cd samples/vision/dinov2/runtime/python
bash run.sh
```

脚本会下载与板端 SoC 匹配的 `.hbm`，对两张测试图推理，打印输出摘要，
并报告两张图 embedding 之间的 cosine 相似度。

## 模型转换

见 [conversion/README_cn.md](./conversion/README_cn.md)。

## 推理

见 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

## 模型评测

见 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 模型列表

| 模型名 | 输入尺寸 | Embedding (cls / patch) | 参数量 | RDK S100 | RDK S100P | RDK S600 |
|---|---|---|---|---|---|---|
| dinov2_vits14_224_int16 | 1x3x224x224 | (1,384) / (1,256,384) | 22.06 M | 3.73 ms / 267.44 FPS | 3.02 ms / 329.53 FPS | 2.25 ms / 441.64 FPS |

## 贡献者

D-Robotics model zoo 团队。

## 许可

源模型与权重为 Apache-2.0 许可的
[DINOv2](https://github.com/facebookresearch/dinov2) 产物。见
[../../../LICENSE](../../../LICENSE)。
