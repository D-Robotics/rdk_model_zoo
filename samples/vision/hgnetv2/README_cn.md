[English](README.md) | 简体中文

# HGNetV2 图像分类

HGNetV2 是用于图像分类的卷积骨干网络。

<a id="overview"></a>

## 概述

HGNetV2 是用于视觉任务的卷积骨干网络；本 sample 提供 b0–b4 的
ImageNet-1k 分类入口。它是面向精度/延迟平衡设计的下一代 CNN 骨干，
是原 HGNet 的后继，在分类、检测与分割任务中均有良好表现。

[PP-HGNetV2](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/models/ImageNet1k/PP-HGNetV2.md)

核心特性：

- **聚合多感受野**——HG-Block 组合由浅至深的多尺度特征，对小型
  目标的检测与识别友好。
- **改进的 stem 模块**——网络入口堆叠更多 2×2 卷积核以学习丰富的
  局部特征，同时使用更小的通道数，提升高分辨率任务表现。
- **可学习下采样（LDS）**——自适应下采样层在降低计算冗余的同时保留
  更多有效空间细节。

输入一张 BGR 图像，输出 ImageNet-1k Top-K 类别 ID、分数和可选标签。
`HGNetV2Classifier` 类执行由 `predict` 串联的
`preprocess → infer → postprocess` 流程（标签读取、绘图和文件输出由
CLI 层负责，见 [runtime/python/README_cn.md](runtime/python/README_cn.md)）。

<a id="directory"></a>
## 目录结构

```text
hgnetv2/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── tests/  # 自动化测试
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── requirements-host.txt  # Python 依赖
```

<a id="support-matrix"></a>
## 支持矩阵

下表列出本 Sample 已发布的模型；`—` 表示该组合未发布。

| 模型 | 输入 | X5 | S100 | S100P | S600 |
| --- | --- | --- | --- | --- | --- |
| HGNetV2-B0 | 224x224 | ✓ | — | — | — |
| HGNetV2-B1 | 224x224 | ✓ | — | — | — |
| HGNetV2-B2 | 224x224 | ✓ | — | — | — |
| HGNetV2-B3 | 224x224 | ✓ | — | — | — |
| HGNetV2-B4 | 224x224 | ✓ | — | — | — |

运行时覆盖：Python CLI 通过 `--target auto|x5` 和
`--variant b0|b1|b2|b3|b4`（默认 `b0`）覆盖全部已发布组合；CLI 小写 ID 映射
到准确的发布文件名 `hgnetv2_b<0-4>_224x224_nv12.bin`（文件名大小写保留）。
`--target s100`、`s100p`、`s600` 不会解析出任何资产。本 Sample 不提供 C++
运行时。

<a id="prerequisites"></a>
## 前提

使用完整仓库检出。X5 推理需要匹配的板端镜像与 `hbm_runtime`；主机
依赖按下方装入虚拟环境。

原生推理不需要 OE 工具链；重新转换的工具链和数据前提见
[conversion](conversion/README_cn.md) 文档。

```bash
# cwd: repository root
python3 -m venv .venv-hgnetv2
source .venv-hgnetv2/bin/activate
python3 -m pip install -r samples/vision/hgnetv2/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速开始

在 X5 仓库根执行。下载成功时退出码 0 并打印观测哈希；推理成功时退出码 0 并打印五条结果。推理不会自动下载模型。

```bash
# cwd: repository root
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/runtime/python/main.py \
  --target x5 --variant b0 \
  --test-img samples/vision/hgnetv2/test_data/sandbar.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="expected-results"></a>
## 预期结果

默认变体为 `b0`；其余变体 `b1`、`b2`、`b3`、`b4` 须显式指定。分数
采用 softmax，完全平局时按 ID 升序稳定排序。`sandbar.JPEG` 用于功能检查；数据集
精度在 ImageNet 验证集上度量。仅指定 `--img-save-path` 才保存文件。

下图为 X5 发布的参考运行效果：demo 将 Top-5 排名叠画在随附的
`sandbar.JPEG` 上，rank 1 为 class 977（sandbar, sand bar）。

![X5 参考推理结果：sandbar 测试图与 Top-5 叠加，
rank 1 为 class 977（sandbar, sand bar）](./test_data/result.jpg)

<a id="performance"></a>
## 性能数据

已发布性能记录：完整列与计时条件见
[评测说明](evaluator/README_cn.md#reference-results)。单线程延迟与多线程 FPS 采用不同的并发方式，
二者不能直接互相取倒数。比较延迟与 FPS 时，应使用相同线程数、并发提交方式和 BPU 利用率。

<a id="entry-points"></a>
## 入口

[Model](model/README_cn.md) · [Python](runtime/python/README_cn.md) · [Conversion](conversion/README_cn.md) · [Evaluation](evaluator/README_cn.md)

<a id="license"></a>
## 许可

Python 代码遵循 Apache-2.0。再分发前请遵循转换材料中的原始声明，并核对上游模型与权重许可。

<!-- converted-batch-20261102 -->
## 已转换新增规格（2026-11-02）

以下 timm 权重单独转换，已通过严格权重加载、结构重参数化及两组输入的
PyTorch/ONNX 主机对照、checker 和编译。板端冒烟、精度、Runtime 1/2 并发和
C++ 1/2 流均为 **not-run**，历史指标和截图不适用于这些产物。原有默认规格保留。
新规格需要 Pillow，使用 bicubic 短边 256、四舍五入居中裁剪 224；Mapper 融合
该权重实际的归一化参数。输出为 float32 1,000 类 logits，由原生 Runtime 做 softmax。

| 规格 | 模型文件 | 权重版本 | 板端 |
|---|---|---|---|
| `timm-b0-ssld-stage2-ft-in1k` | `hgnetv2_b0_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`9e0a904b210d`](https://huggingface.co/timm/hgnetv2_b0.ssld_stage2_ft_in1k/blob/9e0a904b210d65e4fcd6582f8ca6e0921fac2650/README.md) | not-run |
| `timm-b1-ssld-stage2-ft-in1k` | `hgnetv2_b1_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`ec7014d7f1d0`](https://huggingface.co/timm/hgnetv2_b1.ssld_stage2_ft_in1k/blob/ec7014d7f1d0dd1ea2bf3012bc51dc95c5919fd2/README.md) | not-run |
| `timm-b2-ssld-stage2-ft-in1k` | `hgnetv2_b2_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`7a3092bc8711`](https://huggingface.co/timm/hgnetv2_b2.ssld_stage2_ft_in1k/blob/7a3092bc87118c7974bb9000b1eb6d9c560f8f75/README.md) | not-run |
| `timm-b3-ssld-stage2-ft-in1k` | `hgnetv2_b3_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`1aac877b3ea5`](https://huggingface.co/timm/hgnetv2_b3.ssld_stage2_ft_in1k/blob/1aac877b3ea55ab0f06d57cad9889d1e601bd472/README.md) | not-run |
| `timm-b4-ssld-stage2-ft-in1k` | `hgnetv2_b4_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`3c808d9d9d98`](https://huggingface.co/timm/hgnetv2_b4.ssld_stage2_ft_in1k/blob/3c808d9d9d9873ed5e37b00509ba2a4882e816e8/README.md) | not-run |

```bash
python3 samples/vision/hgnetv2/model/download.py --target x5 --variant timm-b0-ssld-stage2-ft-in1k
python3 samples/vision/hgnetv2/runtime/python/main.py --target x5 --variant timm-b0-ssld-stage2-ft-in1k --dry-run
```

精确 URL、SHA-256 和大小登记在 `docs/release/x5/models.yaml`；来源及主机对照
记录见 `conversion/converted.json`。模型许可逐项保留在公开 release 清单中。
<!-- /converted-batch-20261102 -->

<!-- converted-all-20261103 -->

## 本轮 S 系列转换规格（2026-11-03）

本轮新增的独立规格和显式命令见 [模型准备](model/README_cn.md)。板端精度与性能尚未实测。
