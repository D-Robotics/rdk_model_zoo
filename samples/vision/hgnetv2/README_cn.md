# HGNetV2 图像分类

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
└── requirements-host.txt  # 源码或数据文件
```

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | 变体 | Python | C++ |
| --- | --- | --- | --- |
| x5 | b0 | supported | not-supported |
| x5 | b1 | supported | not-supported |
| x5 | b2 | supported | not-supported |
| x5 | b3 | supported | not-supported |
| x5 | b4 | supported | not-supported |
| s100 | b0 | not-supported | not-supported |
| s100 | b1 | not-supported | not-supported |
| s100 | b2 | not-supported | not-supported |
| s100 | b3 | not-supported | not-supported |
| s100 | b4 | not-supported | not-supported |
| s100p | b0 | not-supported | not-supported |
| s100p | b1 | not-supported | not-supported |
| s100p | b2 | not-supported | not-supported |
| s100p | b3 | not-supported | not-supported |
| s100p | b4 | not-supported | not-supported |
| s600 | b0 | not-supported | not-supported |
| s600 | b1 | not-supported | not-supported |
| s600 | b2 | not-supported | not-supported |
| s600 | b3 | not-supported | not-supported |
| s600 | b4 | not-supported | not-supported |

S 系列无对应制品，也按支持矩阵选择 Python 运行时与目标。CLI 小写 ID 映射到准确发布
文件名；文件名大小写保留。

<a id="prerequisites"></a>
## 前提

使用完整仓库检出。X5 推理需要匹配的板端镜像与 `hbm_runtime`；主机
依赖按下方装入虚拟环境。SciPy 仅用于主机对照测试，板端推理不依赖它。

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
